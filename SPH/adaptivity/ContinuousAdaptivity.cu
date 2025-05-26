//#define OLD_STYLE
#define EIGENADAPTIVE

#define OFFSET(v) (0.24509788f * support_from_volume(v) * kernelSize())
//#define OFFSET(v) (0.24509788f * powf(v, 1.f / 3.f)) // 0.6203504908f
//#define INDIVIDUAL_PCA
#include "dummy.h"
#include <SPH/adaptivity/AdaptivityDevice.cuh>
#include <SPH/adaptivity/Blending.cuh>
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <SPH/adaptivity/Merging.cuh>
#include <SPH/adaptivity/MergingBoundary.cuh>
#include <SPH/adaptivity/Sharing.cuh>
#include <SPH/adaptivity/SharingBoundary.cuh>
#include <SPH/adaptivity/Splitting.cuh>
#include <utility/include_all.h>
#include <utility/sdf2.h>


class sort_indices
{
	private:
		int* mparr;
	public:
		sort_indices(int* parr) : mparr(parr) {}
		bool operator()(int i, int j) const { 
			//std::cout << "ij " << i << " " << j << "\n";
			return mparr[i] > mparr[j]; }
};

void pairsort(int a[], int b[], int n)
{
    std::pair<int, int>* pairt;
	pairt = new std::pair<int, int>[n];
    // Storing the respective array
    // elements in pairs.
    for (int i = 0; i < n; i++) 
    {
        pairt[i].first = a[i];
        pairt[i].second = b[i];
    }
 
    // Sorting the pair array.
    std::sort(pairt, pairt + n);
     
    // Modifying original arrays
    for (int i = 0; i < n; i++) 
    {
        a[i] = pairt[i].first;
        b[i] = pairt[i].second;
    }
}
 
struct is_fluid{
			hostDeviceInline bool operator()(int x) { return x == 0; }
		};

basicFunctionType genparticleIndex(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	if (arrays.splitIndicator[i] != 1)
		arrays.particleIndex[i] = INT_MAX;
	else
		arrays.particleIndex[i] = i;
}
basicFunction(indexBlendingParticles, genparticleIndex, "Adaptive: indexing blending particles");

struct is_valid {
	hostDeviceInline bool operator()(const int x) { return x != INT_MAX; }
};
// Main function to call the density blending funciton
void SPH::adaptive::blendDensity(Memory mem) {
	launch<indexBlendingParticles>(mem.num_ptcls, mem);
	get<parameters::adaptive::blendedPtcls>() = (int32_t)algorithm::copy_if(mem.particleIndex, mem.particleIndexCompact, mem.num_ptcls, is_valid());
	launch<blendDensities>(get<parameters::adaptive::blendedPtcls>(), mem, get<parameters::adaptive::blendedPtcls > ());
}

// Main function to call the velocity blending funciton
void SPH::adaptive::blendVelocity(Memory mem) {
	launch<indexBlendingParticles>(mem.num_ptcls, mem);
	get<parameters::adaptive::blendedPtcls>() = (int32_t)algorithm::copy_if(mem.particleIndex, mem.particleIndexCompact, mem.num_ptcls, is_valid());
	launch<blendVelocities>(get<parameters::adaptive::blendedPtcls>(), mem, get<parameters::adaptive::blendedPtcls>());
}
#include <curand_kernel.h>
/** In general for our code basicFunctionType denotes a function that is called on the GPU with no neighborhood
 *information, which are configured via basicFunction to be easier to launch. Similarly neighFunctionType describes a
 *function which requires information about the neighborhood of a particle. deviceInline is just a define to __device__
 *__inline__.
 **/
 // This struct is used to enable the compact operation by creating a thrust compatible functor
struct is_valid_split {
	hostDeviceInline bool operator()(const int x) { return x != INT_MAX; }
};
// this enum is used to make the optimization code more readable by avoiding weakly typed constants.
enum struct threadAssignment : int8_t {
	split, neighbor, none
};
// this function takes a volume as an argument and returns the support radius h based on this value. This value is not
// multiplied by the kernel scale H/h.
deviceInline auto hFromV(float volume) {
	auto target_neighbors = kernelNeighbors();
	auto kernel_epsilon =
		(1.f) * powf((target_neighbors)*PI4O3_1, 1.f / 3.f) / kernelSize();
	auto h = kernel_epsilon * powf(volume, 1.f / 3.f);
	return h;
}
// this function implements an atomic min function for CUDA using CAS as the builtin atomicmin function only works for
// integers and not floats.
__device__ static float atomicMinFloat(float* address, float val) {
	int32_t* address_as_i = (int32_t*)address;
	int32_t old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}
/**
  *This function is intended to classify the particles w.r.t. the splitting operation. Theoretically it should be
  *possible to immedieatly split a particle as soon as it becomes too large, however in practice this does not make
  *sense due to instabilities in the sizing function causing unnecessary fluctuations in spatial resolution. This time
  *delay of 11 timesteps used here is similar to the prior work but not as necessary as there due to the inherently more
  *stable splitting process.
 **/
basicFunctionType indicateParticles(SPH::adaptive::Memory arrays) {
	// This macro returns the index assigned to this thread for this function call for either GPU or CPU execution
	// and returns out of the function if i is larger or equal to the number of threads being run.
	checkedParticleIdx(i);
	// INT_MAX is used to indicate for the compact operation that no splitting is taking place
	arrays.particleIndex[i] = INT_MAX;
	// Particles that do not yet fully exist in the simulation, e.g. those recently split, are exempt from classification
	if (arrays.lifetime[i] < 0.0_s)
		return;
	// adaptiveClassification contains the classification of the particle according to prior work where a positive value
	// indicates that the particle is too large and should be split into n points if the value is n. This value is not
	// clamped to certain ranges
	int32_t decision = static_cast<int32_t>(arrays.adaptiveClassification[i]);
	float decision_b = arrays.adaptiveClassification[i];
	// if the particle can be split into atleast 2 particles
	//printf("|||||||||||descision %f id %d\n", decision_b, i);
	if (decision > 1 || (arrays.particle_type[i] != 0 && decision_b > 1.0f)) {
		// an indicator of 0 is the initial starting value for a particle (zero initialized) and a value of 1 is used to
		// indicate a particle that is being blended. This means that we need to "skip" the 1 value by incrementing twice
		// if the current indicator is 0
		if (arrays.splitIndicator[i] == 0)
			arrays.splitIndicator[i] += 1;
		// the normal increment operation 
		arrays.splitIndicator[i] += 1;
		// which makes sure that a certain number of timesteps have passed since the particle was classified as too large
		// which avoids the aforementioned resolution changes. The threshold of 13 seems to work nicely, although lower 
		// and higher thresholds are a possibility.
		if ((arrays.splitIndicator[i] < 13 && arrays.particle_type[i] == 0) || (arrays.splitIndicator[i] < 3 && arrays.particle_type[i] != 0)) {
			return;
		}
		// the current implementation supports only splitting into up to 16 particles so the value needs to be clamped
		decision = math::clamp(decision, 1, 16);
		if (decision == 16 && arrays.particle_type[i] != 0)
			decision = 1;
		// increment the global counter of particles by the number of particles to insert - 1 as the storage of the 
		// original particle is repurposed
		cuda_atomic<int32_t> num_ptcls(arrays.ptclCounter);
		int32_t split_idx = num_ptcls.add(decision - 1);
		// if the split_idx is too large we revert the global increment atomically and return
		if (split_idx >= arrays.maxNumptcls - decision - 100) {
			num_ptcls.sub(decision - 1);
			return;
		}
		// we store the split_idx in the parent index for now
		arrays.parentIndex.first[i] = split_idx;
		// the particleIndex array is used in compact function to make the overall process significantly faster by 
		// reducing the divergence significantly (of the code). A value != INT_MAX is seen as valid.
		arrays.particleIndex[i] = i;
		//if (arrays.particle_type[i] != 0) printf("_________ParticleIndex uid %d\n", arrays.uid[i]);
	}
	else {
		// if the particle is not classified as too large we reset the splitIndicator to 2 instantly.
		arrays.splitIndicator[i] = 2;
	}
}
/**This function is used to optimize the particles positions using gradient descent and the mass ratios using
 * evolutionary optimization. This function is not called directly but used by another kernel function defined further
 * down. The arguments are:
 * - split_count: the number of particles this particle is split into
 * - X: a shared memory array to store the current positions for GD
 * - gradX: a shared memory array to store the current gradients for GD
 * - tau_s: a shared memory array to store the tau_s value for all split particles
 * - tau_h: a shared memory array to store the tau_h value for all (to an extent) neighboring particles
 * - tau: a shared memory value to store the global minimum refinement error
 * - pIdx: the index of the parent particle, in equations this would be o
 * - sIdx: the index at which the newly refined particles are being inserted into memory
 * - cIdx: the index assigned to a thread from a neighboring particle (H)
 * - sArray: a structure containing pointers to all persistent arrays within the simulation
 * - seed: a random initial value, per block, for the evolutionary optimization
 **/
hostDeviceInline uint32_t xorshift32(uint32_t& state){
	uint32_t x = state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return state = x;
}

deviceInline auto getScalingFactor(SPH::adaptive::Memory& arrays, int32_t i) {
	float pVol = arrays.volume.second[i].val;
	float hl = support_from_volume(pVol);
	float target_neighbors = kernelNeighbors();
	float kernel_epsilon = (1.f) * powf((target_neighbors)*PI4O3_1, 1.f / 3.f) / kernelSize();
	float rl = powf(math::cubic(hl / kernel_epsilon) / PI4O3, 1.f / 3.f);
	rl = powf(pVol / PI4O3, 1.f / 3.f);
	float factor = rl / 0.2714417278766632f * arrays.adaptivityScaling /* * 0.95f*/;
	factor = factor;
	//factor = arrays.adaptivityScaling;
	return hl * kernelSize() * arrays.adaptivityScaling;
}

deviceInline auto getThreshold(SPH::adaptive::Memory& arrays, float volume) {
	return OFFSET(volume) * 1.f * arrays.adaptivityThreshold;
}
#define DBG_STMNT if(false)

template<neighbor_list neigh>
deviceInline void generateInitial(SPH::adaptive::Memory &arrays, int32_t split_count, float3 normal,
	float4* X, float4* gradX, float* tau_s, float* tau_h, float* tau, int32_t pIdx, int32_t sIdx, int32_t cIdx, sortingArray*  sArray, int32_t seed) {
	//constexpr auto neighborhood = neigh;
	__syncthreads();
	if (threadIdx.x == 0) {
		uint32_t state = (uint32_t)((pIdx + sIdx + cIdx) * split_count);
		float u1 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float u2 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float u3 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float4 q = {
			sqrtf(1.f - u1) * sinf(2.f * CUDART_PI_F * u2),
			sqrtf(1.f - u1) * cosf(2.f * CUDART_PI_F * u2),
			sqrtf(u1) * sinf(2.f * CUDART_PI_F * u3),
			sqrtf(u1) * cosf(2.f * CUDART_PI_F * u3)
		};
		gradX[0] = q;
	}
	float twoDFac = (30.f / 7.f) * 1.0;
	// for this function we need to make sure that many of these steps are taken in unity to avoid potential
	// race conditions which would lead to the particle de-synchronizing which breaks their global state that is not
	// explicitly stored but only stored within each local thread which leads to significant problems. As such the usage
	// of __syncthreads() is overly conservative but avoids all potential issues, especially w.r.t. future HW changes
	__syncthreads();
	float thresholdFactor = 1.f * arrays.adaptivityThreshold;// 2.f / 3.f * sqrtf(6.f);
	float pVol = arrays.volume.second[pIdx].val;
	// initially we assume that the ratios are all lambda = 1/n as we split the optimization problems in order to avoid
	// the problems of having to solve large non linear constrained optimization problems.
	float V_s = pVol / (float)split_count;
	float hl = support_from_volume(pVol);
	float target_neighbors = kernelNeighbors();
	float kernel_epsilon = (1.f) * powf((target_neighbors)*PI4O3_1, 1.f / 3.f) / kernelSize();
	float rl = powf(math::cubic(hl / kernel_epsilon) / PI4O3, 1.f / 3.f);
	rl = powf(pVol / PI4O3, 1.f / 3.f);
	float factor = rl / 0.2714417278766632f * arrays.adaptivityScaling /* * 0.95f*/;
	factor = factor;
	factor = getScalingFactor(arrays, pIdx);
	//factor = arrays.adaptivityScaling;
	//factor = hl * kernelSize() / 0.5477225150103355f;
	//factor = 1.f / factor;
	// the factor is used to scale the statically calculated patterns to the actual particle size by calculating
	// the scaling of the support radii which for the static case was 1 with a radius of 0.271...
	auto h = hFromV(V_s);
	if (arrays.particle_type[pIdx] != 0) h = hFromV(V_s / twoDFac);
	// this is the previously described assignment.
	//auto tas = threadAssignment::none;
	float4 assignedPosition;
	//float assignedVolume;

	float3 R1 = {1, 0, 0}, R2 = {0, 1, 0}, R3 = {0, 0, 1};
		
	float rat = (normal.z - 1) / (normal.y * normal.y + normal.x * normal.x);
	if (normal.x != 0 || normal.y != 0)
	{
		R1.x = 1 + normal.x * normal.x * rat;
		R1.y = normal.x * normal.y * rat;
		R1.z = normal.x;
		R2.x = normal.x * normal.y * rat;
		R2.y = 1 + normal.y * normal.y * rat;
		R2.z = normal.y;
		R3.x = -normal.x;
		R3.y = -normal.y;
		R3.z = 1 - (normal.x * normal.x + normal.y * normal.y);
	}
	float4 xtmp;
	
	//printf("r1 %f %f %f r2 %f %f %f r3 %f %f %f\n", R1.x, R1.y, R1.z, R2.x, R2.y, R2.z, R3.x, R3.y, R3.z);

	if (threadIdx.x < split_count) {
		//printf("split cnt %d\n", split_count);
		float4 offset = getshape(split_count, threadIdx.x) * factor;
		if (arrays.particle_type[pIdx] != 0)
			offset = getbshape(split_count, threadIdx.x) * factor;
		float4 v = offset;
		v.w = 0.f;
		float4 u = gradX[0];
		float s = u.w;
		u.w = 0.f;
		// Do the math
		if (arrays.particle_type[pIdx] == 0)
			offset = 2.0f * math::dot3(u, v) * u
			+ (s*s - math::dot3(u, u)) * v
			+ 2.0f * s * math::cross(u, v);

		xtmp.x = math::dot3(offset, R1);
		xtmp.y = math::dot3(offset, R2);
		xtmp.z = math::dot3(offset, R3);

		//printf("normal %f %f %f original %f %f %f rotated %f %f %f\n", normal.x, normal.y, normal.z, offset.x, offset.y, offset.z, xtmp.x, xtmp.y, xtmp.z);

		assignedPosition = arrays.position.second[pIdx].val + xtmp;
		assignedPosition.w = h;
		
		//float threshold = OFFSET(V_s) * thresholdFactor;
		//auto pDist = planeBoundary::distance(uFloat4<SI::m>(assignedPosition), float_u<SI::volume>(V_s), arrays);
		//if (pDist.val.w < threshold)
		//	assignedPosition -= (pDist.val) * (pDist.val.w - threshold);

		//for (int32_t v = 0; v < arrays.volumeBoundaryCounter; ++v) {
		//	auto vDist = volumeBoundary::distance_fn(uFloat4<SI::m>(assignedPosition), float_u<SI::volume>(V_s), arrays, v);
		//	if (vDist.val.w < threshold)
		//		assignedPosition -= (vDist.val) * (vDist.val.w - threshold);
		//}


		X[threadIdx.x] = assignedPosition;
		//X[threadIdx.x].w = h;
		//assignedVolume = V_s;
	}
	else if (cIdx != -1) {
		assignedPosition = arrays.position.second[cIdx].val;
	}
	__syncthreads();
	//return assignedPosition;
}
template<neighbor_list neigh>
deviceInline void gradientDescent(SPH::adaptive::Memory &arrays, int32_t split_count,
	float4* X, float4* gradX, float* tau_s, float* tau_h, float* tau, int32_t pIdx, int32_t cIdx, threadAssignment tas, float pVol, float V_s, float hl, float h) {
	// configuration parameters for GD
	float4 assignedPosition = { 0.f,0.f,0.f,0.f };// getAssignedPosition(arrays, X, cIdx, tas);

	if (tas == threadAssignment::split) {
		assignedPosition = X[threadIdx.x];
	}
	else if (tas == threadAssignment::neighbor) {
		assignedPosition = arrays.position.second[cIdx].val;
	}

	if (tas == threadAssignment::split)
		tau_s[threadIdx.x] = 0.f;
	__syncthreads();
	DBG_STMNT if (blockIdx.x == 0) printf("Entering gradient descent for [%d : %d] @ %d\n", blockIdx.x, threadIdx.x, pIdx);
	__syncthreads();
	constexpr auto neighborhood = neigh;
	float target_neighbors = kernelNeighbors();
	float kernel_epsilon = (1.f) * powf((target_neighbors)*PI4O3_1, 1.f / 3.f) / kernelSize();
	float rl = powf(math::cubic(hl / kernel_epsilon) / PI4O3, 1.f / 3.f);
	rl = powf(pVol / PI4O3, 1.f / 3.f);
	float factor = rl / 0.2714417278766632f * arrays.adaptivityScaling /* * 0.95f*/;
	factor = factor;
	factor = getScalingFactor(arrays, pIdx);
	//factor = arrays.adaptivityScaling;
	auto gamma0 = arrays.adaptivityGamma;
	float thresholdFactor = 1.f * arrays.adaptivityThreshold;// 2.f / 3.f * sqrtf(6.f);
	constexpr auto beta = 0.5f;
	float prevError = FLT_MAX;
	int32_t i = pIdx;
	float gamma = gamma0;
	// these factors can be used to weigh the errors but setting them to 1 works in practice
	constexpr auto hexWeight = 1.f;
	constexpr auto splitWeight = 1.f;
	*tau = 0.f;
	__syncthreads();
	DBG_STMNT if (blockIdx.x == 0 && threadIdx.x == 0)
		printf("%d - [%d : %d] -> factor %e gamma %e beta %e prev %e tau %e\n", __LINE__, blockIdx.x, threadIdx.x, factor, gamma, beta, prevError, *tau);
	__syncthreads();
	DBG_STMNT if (blockIdx.x == 0 && tas == threadAssignment::split) {
		printf("%d - [%d : %d] -> [%f %f %f %f]\n", __LINE__, blockIdx.x, threadIdx.x, X[threadIdx.x].x, X[threadIdx.x].y, X[threadIdx.x].z, X[threadIdx.x].w);
	}
	__syncthreads();
	DBG_STMNT if (blockIdx.x == 0 && tas == threadAssignment::neighbor) {
		printf("%d - [%d : %d] -> [%f %f %f %f] @ %d\n", __LINE__, blockIdx.x, threadIdx.x, assignedPosition.x, assignedPosition.y, assignedPosition.z, assignedPosition.w, cIdx);
	}
	__syncthreads();
	// the code for GD is blocked with an if(true) to easily allow turning this part off.
	if (true) {
		// initially the gradients are set to 0.
		int32_t attempt = 0;
		if (tas == threadAssignment::split) {
			gradX[threadIdx.x] = float4{ 0.f,0.f,0.f,0.f };
		}
		// the actual gradient descent method
		for (int32_t gradientIt = 0; gradientIt < 32 && attempt < 32;) {
			// the following section calculates the error terms tau_s and tau_h in parallel with the equations as they
			// were in the submission.
			if (tas == threadAssignment::neighbor) {
				float tauH = -pVol * kernel(assignedPosition, arrays.position.second[pIdx].val);
				for (int32_t i = 0; i < split_count; ++i) {
					tauH += V_s * kernel(assignedPosition, X[i]);
				}
				tau_h[threadIdx.x - split_count] = tauH * hexWeight;
				// DBG_STMNT if(blockIdx.x == 0) printf("%d - [%d : %d] -> tau_h = %e\n", __LINE__, blockIdx.x, threadIdx.x, tauH);
				atomicAdd(tau, tauH * tauH * hexWeight * hexWeight);
			}
			__syncthreads();
			//float t = *tau;
			__syncthreads();
			float taussib = 0.f;
			float tausnei = 0.f;
			if (tas == threadAssignment::split) {
				float tauS = -arrays.density[pIdx].val;// +
					//boundary::spline(float4_u<SI::m>{assignedPosition}, float_u<SI::volume>{V_s}, arrays.density[pIdx].val, arrays, boundary::kind::both);
				//SWH::spline(float4_u<SI::m>(assignedPosition), arrays).val;
				for (int32_t ia = 0; ia < split_count; ++ia) {
					tauS += V_s * kernel(assignedPosition, X[ia]);
					taussib += V_s * kernel(assignedPosition, X[ia]);
				}
				iterateNeighbors(j) {
					if (j != pIdx) {
						if (arrays.particle_type[pIdx] == 0 || (arrays.particle_type[pIdx] != 0 && arrays.particle_type[j] != 0)) {
							tauS += arrays.volume.second[j].val * kernel(assignedPosition, arrays.position.second[j].val);
							tausnei += arrays.volume.second[j].val * kernel(assignedPosition, arrays.position.second[j].val);
						}
					}
				} 
				// DBG_STMNT if (blockIdx.x == 0) printf("%d - [%d : %d] -> tau_s = %e\n", __LINE__, blockIdx.x, threadIdx.x, tauS);
				//tauS += SWH::spline4(float4_u<SI::m>(assignedPosition), arrays);
				tau_s[threadIdx.x] = tauS * splitWeight;
				atomicAdd(tau, tauS * tauS * splitWeight * splitWeight);
			}
			__syncthreads();
			DBG_STMNT if (blockIdx.x == 0 && threadIdx.x == 0) printf("%d - [%d : %d] -> %e : %e @ [%d : %d] x %e\n", __LINE__, blockIdx.x, threadIdx.x, *tau, prevError, gradientIt, attempt, gamma);
			__syncthreads();
			// we use a very simple condition, which works in practice, to avoid the additional computational complexity
			if (tas == threadAssignment::split && arrays.particle_type[pIdx] != 0/* && arrays.uid[pIdx] == 100*/) {
				//printf("index %d preverr %f dens %f smooth %f neismooth %f\n", pIdx, prevError, arrays.density[pIdx].val, assignedPosition.w, arrays.position.second[arrays.neighborList[i]].val.w);
				/*float tmp = 0.f;
				for (int i = 0; i < split_count; i++)
					tmp += tau_s[i];
				printf("taus %f\n", tmp);
				tmp = 0.f;
				for (int i = split_count; i < 96; i++)
					tmp += tau_h[i];
				printf("tauh %f\n", tmp);*/
		/*		printf("taus %f\n", tau_s[threadIdx.x]);
				printf("taussib %f\n", taussib);
				printf("tausnei %f\n", tausnei); 
				printf("tauh %f\n", tau_h[threadIdx.x]);
		*/	}
			if (*tau < prevError) {
				// synchronize to make sure all threads entered this point then update the global state
				__syncthreads();
				prevError = *tau;
				*tau = 0.f;
				gradientIt++;
				attempt = 0;
				gamma = gamma0;
				__syncthreads();
				// the next section calculates the gradient in parallel as per the submission. This update is only required
				// when a new position is found
				float4 gradH{ 0.f,0.f,0.f,0.f };
				if (tas == threadAssignment::split) {
					float4 grad{ 0.f,0.f,0.f,0.f };
					int32_t ctr = 0;
					iterateNeighbors(j) {
							if (j != pIdx) {
							if (arrays.particle_type[pIdx] == 0 || (arrays.particle_type[pIdx] != 0 && arrays.particle_type[j] != 0)) {
								if (ctr++ >= 96 - split_count) break;
								auto g = arrays.volume.second[j].val * gradient(X[threadIdx.x], arrays.position.second[j].val);
								grad += tau_h[ctr - 1] * g;
								gradH += g;
							}
						}
					}
					gradX[threadIdx.x] = hexWeight * (2.f * grad);
				}
				if (tas == threadAssignment::split) {
					float4 gradA{ 0.f,0.f,0.f,0.f };
					for (int32_t i = 0; i < split_count; ++i) {
						auto g = V_s * gradient(X[threadIdx.x], X[i]);
						gradA += g;
					}
					gradA += V_s * boundary::splineGradient(float4_u<SI::m>{X[threadIdx.x]}, arrays.volume.second[i] * (1.f / (float)split_count), arrays.density[pIdx].val, arrays, boundary::kind::both);

					float4 firstTerm{ 0.f,0.f,0.f,0.f };
					firstTerm = 2.f * tau_s[threadIdx.x] * (gradA + gradH);

					float4 secondTerm{ 0.f,0.f,0.f,0.f };
					for (int32_t i = 0; i < split_count; ++i) {
						if (i != threadIdx.x) {
							secondTerm += -2.f * tau_s[i] * V_s * gradient(X[i], X[threadIdx.x]);
						}
					}
					gradX[threadIdx.x] += firstTerm + secondTerm;
				}
			}
			else {
				// increment the attempt counter after synchronizing and move the global position state back to where it
				// was before and then update gamma. This avoids having to store the actual positions and the potentially
				// updated positions.
				__syncthreads();
				*tau = 0.f;
				attempt++;
				if (tas == threadAssignment::split) {
					X[threadIdx.x] += gradX[threadIdx.x] * gamma;
					X[threadIdx.x].w = assignedPosition.w;
				}
				gamma *= beta;
			}
			__syncthreads();
			// update the positions based on the gradient and step distance
			//float4 previous;
			if (tas == threadAssignment::split) {
				gradX[threadIdx.x].w = 0.f;
				//previous = X[threadIdx.x];
				float threshold = OFFSET(V_s) * thresholdFactor;
				auto pd = math::distance3(X[threadIdx.x], arrays.position.second[pIdx].val);
				if (pd > h * kernelSize())
					X[threadIdx.x] -= (pd - h * kernelSize()) * math::normalize3(X[threadIdx.x] - arrays.position.second[pIdx].val);

				X[threadIdx.x] = X[threadIdx.x] - gradX[threadIdx.x] * gamma;
				//auto pd = math::distance3(X[threadIdx.x], arrays.position.second[pIdx].val);
				//auto t = math::length3(getshape(split_count, threadIdx.x)) * factor * 1.5f;
				//if (pd > t)
				//	X[threadIdx.x] = X[threadIdx.x] - (pd - t) * math::normalize3(X[threadIdx.x] - arrays.position.second[pIdx].val);
				//auto pDist = planeBoundary::distance(uFloat4<SI::m>(X[threadIdx.x]), float_u<SI::volume>(V_s), arrays);
				//if (pDist.val.w < threshold)
				//	X[threadIdx.x] -= (pDist.val) * (pDist.val.w - threshold);
				//for (int32_t v = 0; v < arrays.volumeBoundaryCounter; ++v) {
				//	auto vDist = volumeBoundary::distance_fn(arrays.position.second[i], arrays.volume.second[i], arrays, v);
				//	if (vDist.val.w < threshold)
				//		X[threadIdx.x] -= (vDist.val) * (vDist.val.w - threshold);
				//}
				X[threadIdx.x].w = assignedPosition.w;
				assignedPosition = X[threadIdx.x];
			}
			__syncthreads();
		}
	}
	// This check is required to avoid potential numerical issues with the optimization method.
	if (tas == threadAssignment::split)
		if (X[threadIdx.x].x != X[threadIdx.x].x || X[threadIdx.x].y != X[threadIdx.x].y || X[threadIdx.x].z != X[threadIdx.x].z || X[threadIdx.x].w != X[threadIdx.x].w)
		{
			X[threadIdx.x] = arrays.position.second[pIdx].val + getshape(split_count, threadIdx.x) * factor;
			if (arrays.particle_type[pIdx] != 0)
				X[threadIdx.x] = arrays.position.second[pIdx].val + getbshape(split_count, threadIdx.x) * factor;
		}
	float threshold = OFFSET(V_s) * thresholdFactor *arrays.adaptivityThreshold;
	if (tas == threadAssignment::split) {
		auto pDist = planeBoundary::distance(uFloat4<SI::m>(X[threadIdx.x]), float_u<SI::volume>(V_s), arrays);
		//if (pDist.val.w < threshold)
		//	X[threadIdx.x] -= (pDist.val) * (pDist.val.w - threshold);
		for (int32_t v = 0; v < arrays.volumeBoundaryCounter; ++v) {
			auto vDist = volumeBoundary::distance_fn(arrays.position.second[i], arrays.volume.second[i], arrays, v);
			if (vDist.val.w < threshold)
				X[threadIdx.x] -= (vDist.val) * (vDist.val.w - threshold);
		}
	}
	__syncthreads();

}
deviceInline int32_t emit_particles(SPH::adaptive::Memory &arrays, threadAssignment tas, float pVol, int32_t split_count, float refinementRatio, float3 normal,
	float4* X, float4* gradX, float* tau_s, float* tau_h, float* tau, int32_t pIdx, int32_t sIdx, int32_t cIdx, sortingArray*  sArray, int32_t seed) {
	int32_t new_idx = 0;
	__syncthreads();

	//refinementRatio = math::pow(refinementRatio, 2.0 / 3.0);
	
	//float twoDFac = (30.f / 7.f) * 0.5;
	float twoDFac = 1.0; 
	if (tas == threadAssignment::split) {
		//int32_t split_idx = sIdx;
		//int32_t parent_idx = sIdx + pIdx;
		auto t_0 = -arrays.blendSteps * arrays.timestep;
		//auto x_j = X[threadIdx.x];
		new_idx = threadIdx.x == split_count - 1 ? pIdx : sIdx + threadIdx.x;
		if (split_count == 16 && arrays.particle_type[pIdx] != 0) new_idx = pIdx ;

		// simply copy all properties by default
		sArray->callOnArray([&](auto* first, auto* second) {
			if (first != nullptr)
				first[new_idx] = first[pIdx];
			});
	}
	__syncthreads();
	if (tas == threadAssignment::split) {
		int32_t parent_idx = sIdx + pIdx;
		auto t_0 = -arrays.blendSteps * arrays.timestep;
		auto x_j = X[threadIdx.x];
		
		//printf("poses %f %f %f\n", x_j.x, x_j.y, x_j.z);
		arrays.lifetime[new_idx] = t_0;
		arrays.splitIndicator[new_idx] = 1;
		if (arrays.particle_type[pIdx] == 0)
			arrays.particle_type[new_idx] = 0;
		else arrays.particle_type[new_idx] = arrays.particle_type[pIdx];
		/*if (arrays.tobeoptimized[new_idx] != -1)
		{*/
			arrays.tobeoptimized[new_idx] = 1;
			arrays.optimization_lifetime[new_idx] = 10;
		//}
		if (new_idx != pIdx && arrays.optimization_group[new_idx].val < 0)
		{
			//printf("~~~~~~~~~~~~~~~~MAIN MINUS optgroup %d id %d\n", arrays.optimization_group[new_idx].val, new_idx);
			arrays.optimization_group[new_idx].val -= INT_MIN;
			//printf("~~~~~~~~~~~~~~~~MAIN MINUS1 optgroup %d id %d\n", arrays.optimization_group[new_idx].val, new_idx);
		}
		//else if (new_idx == pIdx) printf("~~~~~~~~~~~~~~~~MAIN optgroup %d id %d\n", arrays.optimization_group[new_idx].val, new_idx);
		//else printf("~~~~~~~~~~~~~~~~MAIN PLUS optgroup %d\n", arrays.optimization_group[new_idx].val);
		arrays.uid[new_idx] = new_idx;
		arrays.parentPosition.first[new_idx] = arrays.position.second[pIdx];
		arrays.parentIndex.first[new_idx] = parent_idx;

		float pSDiv = M_PI * math::pow(3 * pVol / 4 / M_PI, 2.0 / 3.0) / refinementRatio;
		float pVolDiv = math::pow(pSDiv / M_PI, 3.0 / 2.0) * 4 * M_PI / 3;

		arrays.parentVolume[new_idx] = float_u<SI::volume>(pVol);
		if (arrays.particle_type[pIdx] != 0)
			arrays.position.first[new_idx] = float4_u<SI::m>{ x_j.x, x_j.y, x_j.z, hFromV(pVolDiv *  1.f / twoDFac) };
		else arrays.position.first[new_idx] = float4_u<SI::m>{ x_j.x, x_j.y, x_j.z, hFromV(pVol *  1.f / (float)split_count) };
		if (arrays.particle_type[pIdx] == 0)
			arrays.volume.first[new_idx] = float_u<SI::volume>(pVol * 1.f / (float)split_count);
		else arrays.volume.first[new_idx] = float_u<SI::volume>(pVolDiv * 1.f ) / twoDFac;
		arrays.distanceBuffer.first[new_idx] = -3;

		/*if (new_idx == pIdx)
			printf("+++++SMALL VOL %f OLDV %f RAT %f\n", arrays.volume.first[new_idx].val, pVol, refinementRatio);
		*///if (arrays.particle_type[new_idx] != 0) printf("VVVVVVVOL origvol %f curvol %f orig %f cur %f ratio %f\n", arrays.volume.second[pIdx].val, float_u<SI::volume>(pVol * 1.f / (float)refinementRatio).val, hFromV(pVol), arrays.position.first[new_idx].val.w, refinementRatio);
	}
	__syncthreads();
	return new_idx;
}

template<neighbor_list neigh>
deviceInline float optimizeMassDistribution(SPH::adaptive::Memory &arrays, threadAssignment tas, float pVol, int32_t split_count,
	float4* X, float4* gradX, float* tau_s, float* tau_h, float* tau, int32_t pIdx, int32_t sIdx, int32_t cIdx, sortingArray*  sArray, int32_t seed) {
	__syncthreads();
	//return  1.f / (float)split_count;
	int32_t i = pIdx;
	constexpr auto neighborhood = neigh;
	for (int32_t ii = 0; ii < split_count; ++ii)
		tau_s[ii] = 1.f / (float)split_count;
	__syncthreads();
	// The next section executes the actual mass ratio optimization as per the submission
	//if (split_count > 3) {
	//	__syncthreads();
		// initially we assume that the weights are simply lambda = 1/n as the static optimizations do not change the
		// weights significantly.
	for (int32_t ii = 0; ii < split_count; ++ii)
		tau_s[ii] = 1.f / (float)split_count;
	// initialize the curand state for the normal distributions
	curandState localState;
	curand_init(seed + blockIdx.x, threadIdx.x, 0, &localState);
	// configuration parameters
	auto variance = 2.f;
	constexpr auto varianceMultiplier = 0.8f;
	for (int32_t gi = 0; gi < 16; ++gi) {
		// in contrast to before we cannot parallelize the calculation of a single error term but instead
		// calculate the exact same error term on all threads which still works very efficiently due to ideal
		// memory access patterns
		float volumes[16];
		float sum = 0.f;
		// Update the evolution
		for (int32_t ii = 0; ii < split_count; ++ii) {
			// the actual evolution step
			float elem = math::clamp(tau_s[ii] * (float)split_count + curand_normal(&localState) * variance, 0.25f, 2.f);
			if(split_count < 4)
				elem = math::clamp(tau_s[ii] * (float)split_count + curand_normal(&localState) * variance, 0.75f, 1.5f);
			if (threadIdx.x == 0)
				elem = 1.f;
			volumes[ii] = elem;
			sum += elem;
		}
		for (int32_t ii = 0; ii < split_count; ++ii) {
			volumes[ii] /= sum;
		}
		float error = 0.f;
		// calculate the error terms
		iterateNeighbors(j) {
			auto Xs = arrays.position.second[j].val;
			float tauH = -pVol * kernel(Xs, arrays.position.second[i].val);
			for (int32_t ii = 0; ii < split_count; ++ii) {
				auto Xi = float4{ X[ii].x, X[ii].y, X[ii].z, hFromV(pVol * volumes[ii]) };
				tauH += pVol * volumes[ii] * kernel(Xs, Xi);
			}
			//tauH /= (float)arrays.neighborListLength[i];
			error += tauH * tauH;
		}
		//float tauB = -pVol * boundary::spline(arrays.position.second[i], arrays.volume.second[i], arrays, boundary::kind::plane, -1).val;
		//for (int32_t ii = 0; ii < split_count; ++ii) {
		//	auto Xi = float4{ X[ii].x, X[ii].y, X[ii].z, hFromV(pVol * volumes[ii]) };
		//	tauB += pVol * volumes[ii] * boundary::spline(Xi, arrays.volume.second[i], arrays, boundary::kind::plane, -1).val;
		//}
		//error += tauB * tauB;

		//float tauS0 = -arrays.density[i].val +
		//	 boundary::spline(arrays.position.first[i], pVol * volumes[ii], arrays);
			//SWH::spline(arrays.position.first[i], arrays).val;
		for (int32_t ii = 0; ii < split_count; ++ii) {
			auto Xi = float4{ X[ii].x, X[ii].y, X[ii].z, hFromV(pVol * volumes[ii]) };
			float tauS = -arrays.density[i].val + boundary::spline(Xi, pVol * volumes[ii], 0.5f, arrays, boundary::kind::both).val;
			for (int32_t ia = 0; ia < split_count; ++ia) {
				auto Xs = float4{ X[ia].x, X[ia].y, X[ia].z, hFromV(pVol * volumes[ia]) };
				tauS += pVol * volumes[ia] * kernel(Xs, Xi);
			}
			iterateNeighbors(j) {
				if (j != pIdx) {
					tauS += arrays.volume.second[j].val * kernel(Xi, arrays.position.second[j].val);
				}
			}
			//tauS /= (float)split_count;
			error += tauS * tauS;
		}
		// we now pick the term with the lowest error using an atomic operation and use that threads properties
		// to update the global state stored in tau_s
		*tau = FLT_MAX;
		__syncthreads();
		atomicMinFloat(tau, error);
		__syncthreads();
		//if(threadIdx.x == 0){
		if (atomicCAS((int32_t*)tau, __float_as_int(error), __float_as_int(-error)) == __float_as_int(error)) {
			for (int32_t ii = 0; ii < split_count; ++ii)
				tau_s[ii] = volumes[ii];
		}
		//if(tas == threadAssignment::split) arrays.debugArray[new_idx].x = error;
		// update the optimizer
		//__syncthreads(); 
		//if (blockIdx.x == 0 && threadIdx.x == 0) {
		//	printf("##################################################\n%f : %f @ %f[%d] -> \n[%f %f %f %f |\n %f %f %f %f |\n %f %f %f %f |\n %f %f %f %f]\n",
		//		error, *tau, variance, gi,
		//		split_count < 1 ? 0.f : tau_s[0], split_count < 2 ? 0.f : tau_s[1], split_count < 3 ? 0.f : tau_s[2], split_count < 4 ? 0.f : tau_s[3],
		//		split_count < 5 ? 0.f : tau_s[4], split_count < 6 ? 0.f : tau_s[5], split_count < 7 ? 0.f : tau_s[6], split_count < 8 ? 0.f : tau_s[7],
		//		split_count < 9 ? 0.f : tau_s[8], split_count < 10 ? 0.f : tau_s[9], split_count < 11 ? 0.f : tau_s[10], split_count < 12 ? 0.f : tau_s[11],
		//		split_count < 13 ? 0.f : tau_s[12], split_count < 14 ? 0.f : tau_s[13], split_count < 15 ? 0.f : tau_s[14], split_count < 16 ? 0.f : tau_s[15]);
		//}
		__syncthreads();
		variance *= varianceMultiplier;
	}
	__syncthreads();
	//}
	if (tas != threadAssignment::split)
		return 0.f;
	// all non split assigned particles are now done and all that is left to do is to create the actual refined particles
	if (threadIdx.x == 0) {
		float sum = 0.f;
		bool flag = false;
		for (int32_t ii = 0; ii < split_count; ++ii) {
			sum += tau_s[ii];
			flag = tau_s[ii] <= 0.f || (tau_s[ii] != tau_s[ii]) || flag;
		}
		if (flag) {
			for (int32_t ii = 0; ii < split_count; ++ii)
				tau_s[ii] = 1.f;
			sum = split_count;

			for (int32_t ii = 0; ii < split_count; ++ii) {
				tau_s[ii] /= sum;
			}
		}
	}
	//__syncthreads();
	// 0.469850
	// 0.24509788044034622
	return tau_s[threadIdx.x];

}
deviceInline auto getAssignment(int32_t split_count, int32_t cIdx) {
	if (threadIdx.x < split_count) {
		return threadAssignment::split;
	}
	else if (cIdx != -1) {
		return threadAssignment::neighbor;
	}
	return threadAssignment::none;
}
deviceInline auto getAssignedPosition(const SPH::adaptive::Memory& arrays, float4* X, int32_t cIdx, threadAssignment tas) {
	if (tas == threadAssignment::split) {
		return X[threadIdx.x];
	}
	else if (tas == threadAssignment::neighbor) {
		return arrays.position.second[cIdx].val;
	}
	return float4{ 0.f,0.f,0.f,0.f };
}

deviceInline bool parentCloseToBoundary(SPH::adaptive::Memory& arrays, int32_t i) {
	auto pDist = planeBoundary::distance(arrays.position.second[i], arrays.volume.second[i], arrays);
	if (pDist.val.w < support_from_volume(1.f) * kernelSize() * 1.5f)
		return true;
	for (int32_t v = 0; v < arrays.volumeBoundaryCounter; ++v)
		if (volumeBoundary::distance_fn(arrays.position.second[i], arrays.volume.second[i], arrays, v) < support_from_volume(1.f) * kernelSize() * 1.5f)
			return true;
	return false;
}

template<neighbor_list neighborhood>
deviceInline void randomizePattern(SPH::adaptive::Memory& arrays, int32_t split_count, float4* X, float4* refX, float* result, int32_t i, uint32_t state) {
	float pVol = arrays.volume.second[i].val;
	float4 pPos = arrays.position.second[i].val;
	float V_s = pVol / (float)split_count;
	if (threadIdx.x < split_count) {
		refX[threadIdx.x] = getshape(split_count, threadIdx.x) * getScalingFactor(arrays, i);
		if (arrays.particle_type[i] != 0)
			refX[threadIdx.x] = getbshape(split_count, threadIdx.x) * getScalingFactor(arrays, i);
		refX[threadIdx.x].w = 0.f;
	}
	__syncthreads();
	state += threadIdx.x;
	*result = -FLT_MAX;
	__syncthreads();
	for (int32_t j = 0; j < 8; ++j) {
		if (threadIdx.x == 0)
			*result = -*result;
		__syncthreads();

		float u1 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float u2 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float u3 = (float)(xorshift32(state) & ((1 << 24) - 1)) / (float)(1 << 24);
		float4 q = {
			sqrtf(1.f - u1) * sinf(2.f * CUDART_PI_F * u2),
			sqrtf(1.f - u1) * cosf(2.f * CUDART_PI_F * u2),
			sqrtf(u1) * sinf(2.f * CUDART_PI_F * u3),
			sqrtf(u1) * cosf(2.f * CUDART_PI_F * u3)
		};
		auto getNew = [&](int32_t idx) {
			float4 v = refX[idx];
			v.w = 0.f;
			float4 u = q;
			float s = u.w;
			u.w = 0.f;
			float4 offset = 2.0f * math::dot3(u, v) * u + (s*s - math::dot3(u, u)) * v + 2.0f * s * math::cross(u, v);
			float4 result = pPos + offset;
			result.w = support_from_volume(V_s);
			return result;
		};
		float densityS[17];
		for (int32_t s = 0; s < split_count + 1; ++s)
			densityS[s] = 0.f;
		densityS[16] = pVol * kernel(pPos, pPos) + boundary::spline(pPos, pVol, arrays.density[i], arrays, boundary::kind::both);
		__syncthreads();
		float tau = 0.f;
		iterateNeighbors(j) {
			auto x_j = arrays.position.second[j].val;
			auto V_j = arrays.volume.second[j].val;
			auto kp = kernel(x_j, pPos);
			float tauj = -pVol * kp;
			densityS[16] += V_j * kp;
			for (int32_t s = 0; s < split_count; ++s) {
				auto ks = kernel(x_j, getNew(s));
				tauj += V_s * ks;
				densityS[s] += V_j * ks;
			}
			tau += tauj * tauj;
		}
		for (int32_t s = 0; s < split_count; ++s) {
			auto xs = getNew(s);
			for (int32_t ss = 0; ss < split_count; ++ss)
				densityS[s] += V_s * kernel(xs, getNew(ss));
		}
		float tauS = 0.f;
		for (int32_t s = 0; s < split_count; ++s) {
			densityS[s] = densityS[16] - densityS[s] - boundary::spline(getNew(s), V_s, densityS[s], arrays, boundary::kind::both);
			tauS += densityS[s] * densityS[s];
		}
		tau += tauS;
		__syncthreads();
		atomicMinFloat(result, tau);
		__syncthreads();
		if (atomicCAS((int32_t*)result, __float_as_int(tau), __float_as_int(-tau)) == __float_as_int(tau))
			for (int32_t s = 0; s < split_count; ++s)
				X[s] = getNew(s);
		__syncthreads();
	}
}

template<neighbor_list neigh>
deviceInline void gradient_particles(SPH::adaptive::Memory &arrays, int32_t split_count, float refinementRatio, float3 normal,
	float4* X, float4* gradX, float* tau_s, float* tau_h, float* tau, int32_t pIdx, int32_t sIdx, int32_t cIdx, sortingArray*  sArray, int32_t seed) {
	*tau = 0.f;
	//constexpr auto neighborhood = neigh;
	//int32_t i = pIdx;
	float twoDFac = (30.f / 7.f) * 1.0;

	float pVol = arrays.volume.second[pIdx].val;
	float hl = support_from_volume(pVol);	
	float V_s = pVol / (float)split_count;
	if (arrays.particle_type[pIdx] != 0) V_s /= twoDFac;
	auto h = hFromV(V_s);

	auto tas = getAssignment(split_count, cIdx);
	//printf("type %d\n", arrays.particle_type[pIdx]);
	generateInitial<neigh>(arrays, split_count, normal, X, gradX, tau_s, tau_h, tau, pIdx, sIdx, cIdx, sArray, seed);
	if (arrays.particle_type[pIdx] == 0)
		randomizePattern<neigh>(arrays, split_count, X, gradX, tau, pIdx, (uint32_t)((pIdx + sIdx + cIdx + threadIdx.x) * split_count));
	gradientDescent<neigh>(arrays, split_count, X, gradX, tau_s, tau_h, tau, pIdx, cIdx, tas, pVol, V_s, hl, h);
	
	int32_t new_idx = emit_particles(arrays,tas, pVol, split_count, refinementRatio, normal, X, gradX, tau_s, tau_h, tau, pIdx, sIdx, cIdx, sArray, seed);
	//if (parentCloseToBoundary(arrays,i) || (split_count < 5) )
	//	return;
	//if (split_count < 4) return;
	if (arrays.particle_type[pIdx] == 0 || arrays.particle_type[new_idx] == 0 /*|| true*/)
	{
		auto ratio = optimizeMassDistribution<neigh>(arrays, tas, pVol, split_count, X, gradX, tau_s, tau_h, tau, pIdx, sIdx, cIdx, sArray, seed);
		if (tas == threadAssignment::split &&  arrays.particle_type[new_idx] == 0) {
			math::unit_assign<4>(arrays.position.first[new_idx], float_u<SI::m>(hFromV(pVol * ratio)));
			arrays.volume.first[new_idx] = float_u<SI::volume>(pVol * ratio);
		}
	}
	/*auto a = arrays.position.first[new_idx].val;
	if (new_idx != 0)
		printf("id %d type %d newpos %f %f %f\n", new_idx, arrays.particle_type[new_idx], a.x, a.y, a.z);*/


}
/** This function is used to initialize the state for the dynamic optimization by declaring
 * all reuqired shared memory arrays, as well as assigning neighboring particles to threads of a block. This
 * implementation in general works by assigning n = 96 threads to a particle and then splitting the thread assginments.
 * The first s threads, with s being the number of particles a particle is split into, are assigned to handle the split
 * particles for the error function evaluation, wheras the other threads arae  initially assigned to a separate
 * neighboring particle. If the number of neighbors is larger than the number of threads we ignore this point. In order
 * to avoid potential kernel time outs, especially on windows, this function is inteded to be called with a limited
 * number of blocks, where each block is still indexed from 1 to this number based on the way blockIdx works. As such we
 * pass the offset of the current call to this function to be able to calculate the actual index. It is then
 * straight-forward to calculate the actual index using the result of the compact operation. ptcls is the number of
 * particles that are being split, seed is used as the seed for the evolutionary optimization. sArray is just passed.**/
 /* The state word must be initialized to non-zero */
template<neighbor_list neighborhood>
__global__ void gradientSplit(SPH::adaptive::Memory arrays, int32_t offset, int32_t ptcls, int32_t seed, sortingArray* sArray) {
	// the global shared memory state. In total 32 float4 elements and 113 float elements are used for a total of
	// 964 Byte of shared memory per block. With most gpu architectures 64KB of SM are available per SMX which means
	// that the limit of blocks based on SM with a block size of 96 is 67 which is larger than the theoretical HW limit
	// of 21 so no limit to occupancy exists here.
	__shared__ float4 xn[16];
	__shared__ float4 gradXn[16];
	__shared__ float tau_s[16];
	__shared__ float tau_h[96];
	__shared__ float tau[1];
	for (int i = 0; i < 96; i++)
		tau_h[i] = 0.f;
	// first we get the actual index of the particle by looking at the result of the compaction
	DBG_STMNT if (blockIdx.x != 0) return;
	DBG_STMNT	printf("%d - [%d : %d] -> %d : %d @ %d\n", __LINE__, blockIdx.x, threadIdx.x, offset, offset + blockIdx.x, ptcls);
	int32_t idx = blockIdx.x + offset;
	if (idx >= ptcls) return;
	int32_t i = arrays.particleIndexCompact[idx];
	// and load all required information, as well as reclamping the decision for safety
	int32_t decision = arrays.adaptiveClassification[i];
	int32_t splitIdx = arrays.parentIndex.first[i];
	//printf("index %d type %d spltidxc %d\n", i, arrays.particle_type[i], splitIdx);

	int32_t careTakerIdx = -1;
	decision = math::clamp(decision, 2, 16);
	if (arrays.particle_type[i] != 0 && decision > 8 && decision < 16) decision = 8;
	//if (decision == 3)
	//	decision = 4;
	// this synchthreads is not strictly required but helps conceptually make the code more clear
	__syncthreads();
	// for all threads > s, as described before, we assign one neighboring particle. This works by
	// iterating over all neighbors of o and assigning the c-th neighbor to the c + s-th thread. This
	// could be significantly simplified when using certain neighbor list algorithms.
	if (threadIdx.x >= decision) {
		int32_t ctr = 0;
		// we always use neighbor_list::constrained for adaptive simulations.
		//constexpr auto neighborhood = neighbor_list::cell_based;
		iterateNeighbors(j) {
			// we do not assign the original particle o to any thread
			if (j == i) continue;
			auto k = kernel(arrays.position.second[i], arrays.position.second[j]);
			if (k > 0.f) {
				if (ctr + decision == threadIdx.x)
					careTakerIdx = j;
				ctr++;
			}
		}
	}
	if (threadIdx.x == 0)
		atomicAdd(arrays.adaptivityCounter + (math::clamp(decision, 2, 16) - 1), 1);
	__syncthreads();

	if (threadIdx.x != 0 && decision == 16 && arrays.particle_type[i] != 0) return;
	if (threadIdx.x < decision) {
		//printf("splitting %d type %d class %f decis %d\n", i, arrays.particle_type[i], arrays.adaptiveClassification[i], decision);
	}
	

	float3 currentPoint = {arrays.position.first[i].val.x, arrays.position.first[i].val.y, arrays.position.first[i].val.z};
	// IT IS TRAP IN RESOLUTION. IT SHOULD BE +20 MORE. SEE SDF.H CODE
	if (arrays.particle_type[i] != 0)
	{
		int trueResolution = arrays.sdf_resolution + 20;
		float* grd = (float*)arrays.rigidbody_sdf + trueResolution*trueResolution*trueResolution*(arrays.particle_type[i] - 1);
		
		float signedDistance = sdf::lookupSDF(grd, currentPoint, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]);
		float3 gradient{
			(math::abs(sdf::lookupSDF(grd, currentPoint + float3{ arrays.sdf_epsilon, 0, 0 }, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1])) - math::abs(signedDistance)) / arrays.sdf_epsilon,
			(math::abs(sdf::lookupSDF(grd, currentPoint + float3{ 0, arrays.sdf_epsilon, 0 }, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1])) - math::abs(signedDistance)) / arrays.sdf_epsilon,
			(math::abs(sdf::lookupSDF(grd, currentPoint + float3{0, 0, arrays.sdf_epsilon}, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1])) - math::abs(signedDistance)) / arrays.sdf_epsilon
		};
		gradient /= math::length3(gradient);
		//printf("grad/normal %f %f %f\n", gradient.x, gradient.y, gradient.z);
		//printf("id %d thread %d des %d type %d\n", i, threadIdx.x, decision, arrays.particle_type[i]);
		//return;
		// call the actual function that does the dynamic optimizations.
		gradient_particles<neighborhood>(arrays, decision, arrays.refinementRatio[i], gradient, xn, gradXn, tau_s, tau_h, tau, i, splitIdx, careTakerIdx, sArray, seed);
	}
	else 
		gradient_particles<neighborhood>(arrays, decision, arrays.refinementRatio[i], {0.0, 0.0, 0.0}, xn, gradXn, tau_s, tau_h, tau, i, splitIdx, careTakerIdx, sArray, seed);
}
// This wrapper is there to simplify calling certain functions and wraps indicateParticles into splitIndices
basicFunction(splitIndices, indicateParticles, "Adaptive: splitting particles");
// The main function of the splitting process on host side
void splittingParticles(SPH::adaptive::Memory mem) {
	// decide is a function that classifies the particles using some user defined method, e.g. using surface distance
	// which is use-case specific. The results are stored in adaptiveClassification as floats.
	float velmax = -1.f;
	float velmin = -1.f;

	for (int i = 0; i < mem.num_ptcls; i++)
	{
		if (mem.particle_type[i] != 0) continue;
		float vel = math::length3(mem.velocity[i].val);
		if (velmin == -1.f) velmin = vel;
		if (velmax == -1.f) velmax = vel;
		if (vel > velmax) velmax = vel;
		if (vel < velmin) velmin = vel;
	}
	mem.max_vel = velmax;
	mem.min_vel = velmin;
	cuda::sync();
	std::cout << "MAMIN " << mem.max_vel << " " << mem.min_vel << std::endl;
	launch<decide>(mem.num_ptcls, mem);
	launch<initArrayForDec>(mem.num_ptcls, mem);
	launch<decideRefinement>(mem.num_ptcls, mem);
	std::cout << "MAMIN2 " << mem.max_vel << " " << mem.min_vel << std::endl;
	
#ifndef EIGENADAPTIVE
	int32_t old = mem.num_ptcls;
	cuda::memcpy(mem.ptclCounter, &mem.num_ptcls, sizeof(int32_t), cudaMemcpyHostToDevice);
	callSplit(sorting_list, mem);
	cuda::memcpy(&mem.num_ptcls, mem.ptclCounter, sizeof(int32_t), cudaMemcpyDeviceToHost);
	get<parameters::internal::num_ptcls>() = mem.num_ptcls;
	get<parameters::adaptive::splitPtcls>()[1] = mem.num_ptcls - old;
#else
	std::cout << "MAMIN211111 " << mem.ptclCounter << " " << *mem.ptclCounter << " " << mem.num_ptcls << std::endl;
	// we use a simple static set of variables to calculate the seeds using C++11 random functions.
	// store the old number of particles to calculate the number of changed entries for visualization
	//int32_t old = mem.num_ptcls;
	// store the number of particles into a global to be atomically incremented/decremented
	cuda::memcpy(mem.ptclCounter, &mem.num_ptcls, sizeof(int32_t), cudaMemcpyHostToDevice);
	std::cout << "MAMIN3 " << mem.max_vel << " " << mem.min_vel << std::endl;
	// indicate the particles by either storing INT_MAX (invalid) or their index(valid) in particleIndex
	launch<splitIndices>(mem.num_ptcls, mem);
	std::cout << "MAMIN4 " << mem.max_vel << " " << mem.min_vel << std::endl;
	// we compact particleIndex which gives the number of particles that are being split.
	int32_t ptcls = (int32_t)algorithm::copy_if(mem.particleIndex, mem.particleIndexCompact, mem.num_ptcls, is_valid_split());
	printf("PTCLS %d\n", ptcls);
	if (ptcls > 0) {
		// we store the old positions in the rear pointers of the arrays as the actual values change in the split process
		// which also allows us to not have to recalculate particle neighbors after every split operation. However, this
		// is similar in concept to assuming that all split operations are independently optimized which leads to the 
		// errors as described in the submission.
		cuda::memcpy(arrays::position::rear_ptr, arrays::position::ptr, mem.num_ptcls * sizeof(float4), cudaMemcpyDeviceToDevice);
		cuda::memcpy(arrays::volume::rear_ptr, arrays::volume::ptr, mem.num_ptcls * sizeof(float), cudaMemcpyDeviceToDevice);
		// sortingArray contains an entry per persistent array that needs to be touched by the splitting process. The 
		// fillArray function assigns the correct pointers to all entries.
		sortingArray sArray{};
		static sortingArray* sArrayPtr;
		static bool once = true;
		if (once) {
			cudaMalloc(&sArrayPtr, sizeof(sortingArray));
			once = false;
		}
		sArray.fillArray();
		cudaMemcpy(sArrayPtr, &sArray, sizeof(sortingArray), cudaMemcpyHostToDevice);
		int32_t stepSize = 768*2;
		// this incremental call is intended to avoid kernel launch time outs on windows.
		
		for (int32_t i = 0; i <= ptcls / stepSize; ++i) {
			switch (get<parameters::internal::neighborhood_kind>()) {
			case neighbor_list::cell_based:
				gradientSplit<neighbor_list::cell_based> <<<stepSize, 96 >>> (mem, i * stepSize, ptcls, get<parameters::internal::frame>() + i, sArrayPtr);
				break;
			case neighbor_list::compactCell:
				gradientSplit<neighbor_list::compactCell> <<<stepSize, 96 >>> (mem, i * stepSize, ptcls, get<parameters::internal::frame>() + i, sArrayPtr);
				break;
			case neighbor_list::constrained:
				gradientSplit<neighbor_list::constrained> <<<stepSize, 96 >>> (mem, i * stepSize, ptcls, get<parameters::internal::frame>() + i, sArrayPtr);
				break;
			case neighbor_list::compactMLM:
				gradientSplit<neighbor_list::compactMLM> <<<stepSize, 96 >>> (mem, i * stepSize, ptcls, get<parameters::internal::frame>() + i, sArrayPtr);
				break;
			case neighbor_list::masked:
				gradientSplit<neighbor_list::masked> <<<stepSize, 96 >>> (mem, i * stepSize, ptcls, get<parameters::internal::frame>() + i, sArrayPtr);
				break;
			}
			cuda::sync();
			//break;
		}
		// we read back the current state of the atomic counter in order to update the number of particles.
		cuda::memcpy(&mem.num_ptcls, mem.ptclCounter, sizeof(int32_t), cudaMemcpyDeviceToHost);
	}
	cuda::sync();
	// update global parameters based on the result of the splitting process.
	get<parameters::internal::num_ptcls>() = mem.num_ptcls;
	//get<parameters::adaptive::splitPtcls>() = mem.num_ptcls - old;
#endif
} 
#include <random>


/** This function is used to adjust the resolution of particles in the simulation it does splitting,
 * merging and mass sharing closely following the reference paper**/
//is_fluid{
//			hostDeviceInline bool operator()(int x) { return x == 0; }
//		}


void SPH::adaptive::adapt(Memory mem) {

	static bool once = true;
	if (once) {
		//std::cout << "Continuous Adaptivity module built " << __TIMESTAMP__ << std::endl;

		get<parameters::adaptive::splitPtcls>().resize(16);
		get<parameters::adaptive::mergedPtcls>().resize(16);
		get<parameters::adaptive::sharedPtcls>().resize(16);
		once = false;
	}
	uint32_t split_ptcls;
	/* To avoid certain extra checks in the code we can restrict the function to either merge
	 particles (decreasing resolution) or to split particles (increasing resolution). As this is
	 done on a 2 frame period this should have no appreciable effect on the adaptation rate.*/
	std::cout << "________BEGIN ADAPT limit " << mem.surface_levelLimit.val << "\n";
	if (get<parameters::internal::frame>() % 3 == 0) {
		//return;
		launch<decide>(mem.num_ptcls, mem);
		launch<initArrayForDec>(mem.num_ptcls, mem);
		launch<decideRefinement>(mem.num_ptcls, mem);
		cuda::Memset(mem.adaptivityCounter, 0x00, sizeof(int32_t) * 16);
		cuda::Memset(mem.mergeCounter, 0x00, sizeof(float) * mem.num_ptcls);
		cuda::Memset(mem.ptclCounter, 0x00, sizeof(float));
		cuda::Memset(mem.mergeable, 0x00, sizeof(uint32_t) * mem.num_ptcls);
		std::random_device rngd;
		auto rng = std::default_random_engine{rngd()};
		std::vector<int32_t> idxs(mem.num_ptcls);
		std::iota(std::begin(idxs), std::end(idxs), 0);
		std::shuffle(std::begin(idxs), std::end(idxs), rng);
		int32_t* randomIdx = nullptr;
		cudaMalloc(&randomIdx, sizeof(int32_t) * mem.num_ptcls);
		cudaMemcpy(randomIdx, idxs.data(), sizeof(int32_t) * mem.num_ptcls, cudaMemcpyHostToDevice);

		launch<detectMergingParticles>(mem.num_ptcls, mem);
		cuda::sync();
		launch<grabEvenMergingParticles>(mem.num_ptcls, mem, randomIdx);
		cuda::sync();
		launch<grabOddMergingParticles>(mem.num_ptcls, mem, randomIdx);
		cuda::sync();
		cudaFree(randomIdx);
		
		
		MergeGrabbed(sorting_list, mem);

		
		cuda::memcpy(get<parameters::adaptive::mergedPtcls>().data(), mem.adaptivityCounter, sizeof(int32_t) * 16, cudaMemcpyDeviceToHost);
		cuda::memcpy(&split_ptcls, mem.ptclCounter, sizeof(uint), cudaMemcpyDeviceToHost);
		//get<parameters::adaptive::mergedPtcls>() = split_ptcls;
	}
	else if (get<parameters::internal::frame>() % 3 == 1) {
		return;
		launch<decide>(mem.num_ptcls, mem);
		launch<initArrayForDec>(mem.num_ptcls, mem);
		launch<decideRefinement>(mem.num_ptcls, mem);
		cuda::Memset(mem.adaptivityCounter, 0x00, sizeof(int32_t) * 16);
		cuda::Memset(mem.mergeCounter, 0x00, sizeof(float) * mem.num_ptcls);
		cuda::Memset(mem.ptclCounter, 0x00, sizeof(float));
		cuda::Memset(mem.mergeable, 0x00, sizeof(uint32_t) * mem.num_ptcls);
		std::random_device rngd;
		auto rng = std::default_random_engine{rngd()};
		std::vector<int32_t> idxs(mem.num_ptcls);
		std::iota(std::begin(idxs), std::end(idxs), 0);
		std::shuffle(std::begin(idxs), std::end(idxs), rng);
		int32_t* randomIdx = nullptr;
		cudaMalloc(&randomIdx, sizeof(int32_t) * mem.num_ptcls);
		cudaMemcpy(randomIdx, idxs.data(), sizeof(int32_t) * mem.num_ptcls, cudaMemcpyHostToDevice);

		///BOUNDARY MERGING
			launch<detectMergingParticlesBoundary>(mem.num_ptcls, mem);
		cuda::sync();
		launch<grabEvenMergingParticlesBoundary>(mem.num_ptcls, mem, randomIdx);
		cuda::sync();
		launch<grabOddMergingParticlesBoundary>(mem.num_ptcls, mem, randomIdx);
		cuda::sync();
		cudaFree(randomIdx);
		
		MergeGrabbedBoundary(sorting_list, mem);

		cuda::memcpy(get<parameters::adaptive::mergedPtcls>().data(), mem.adaptivityCounter, sizeof(int32_t) * 16, cudaMemcpyDeviceToHost);
		cuda::memcpy(&split_ptcls, mem.ptclCounter, sizeof(uint), cudaMemcpyDeviceToHost);
		//get<parameters::adaptive::mergedPtcls>() = split_ptcls;
	}
	else if (get<parameters::internal::frame>() % 3 == 2){
		//std::cout << "..." << std::endl;
		// Share particles
		launch<decide>(mem.num_ptcls, mem);
		launch<initArrayForDec>(mem.num_ptcls, mem);
		launch<decideRefinement>(mem.num_ptcls, mem);
		cuda::Memset(mem.adaptivityCounter, 0x00, sizeof(int32_t) * 16);
		cuda::Memset(mem.ptclCounter, 0x00, sizeof(float));
		cuda::Memset(mem.mergeable, 0x00, sizeof(uint32_t) * mem.num_ptcls);
		cuda::Memset(mem.mergeCounter, 0x00, sizeof(float) * mem.num_ptcls);
		std::random_device rngd;
		auto rng = std::default_random_engine{ rngd() };
		std::vector<int32_t> idxs(mem.num_ptcls);
		std::iota(std::begin(idxs), std::end(idxs), 0);
		std::shuffle(std::begin(idxs), std::end(idxs), rng);
		int32_t* randomIdx = nullptr;
		cudaMalloc(&randomIdx, sizeof(int32_t) * mem.num_ptcls);
		cudaMemcpy(randomIdx, idxs.data(), sizeof(int32_t) * mem.num_ptcls, cudaMemcpyHostToDevice);
		launch<detectSharingParticles>(mem.num_ptcls, mem);
		launch<grabEvenSharingParticles>(mem.num_ptcls, mem, randomIdx);
		launch<grabOddSharingParticles>(mem.num_ptcls, mem, randomIdx);
		
		////sharing for boundary
		/*launch<detectSharingParticlesBoundary>(mem.num_ptcls, mem);
		launch<grabEvenSharingParticlesBoundary>(mem.num_ptcls, mem, randomIdx);
		launch<grabOddSharingParticlesBoundary>(mem.num_ptcls, mem, randomIdx);
		*/cudaFree(randomIdx);
		
		ShareGrabbed(sorting_list, mem);
		//ShareGrabbedBoundary(sorting_list, mem);
		
		cuda::memcpy(&split_ptcls, mem.ptclCounter, sizeof(uint), cudaMemcpyDeviceToHost);
		cuda::memcpy(get<parameters::adaptive::sharedPtcls>().data(), mem.adaptivityCounter, sizeof(int32_t) * 16, cudaMemcpyDeviceToHost);
		//get<parameters::adaptive::sharedPtcls>() = split_ptcls;
		// Split particles, if the old particle count is close to the maximum particle count of the
		// simulation do nothing.

		cuda::Memset(mem.adaptivityCounter, 0x00, sizeof(int32_t) * 16);
		cuda::sync();
		std::cout << "!!!!!!!! BEFORE SPLITTING " << std::endl;
		splittingParticles(mem);
		cuda::sync();
		cuda::memcpy(get<parameters::adaptive::splitPtcls>().data(), mem.adaptivityCounter, sizeof(int32_t) * 16, cudaMemcpyDeviceToHost);
		//int32_t old = mem.num_ptcls;
		//cuda::memcpy(mem.ptclCounter, &mem.num_ptcls, sizeof(int32_t), cudaMemcpyHostToDevice);
		//callSplit(sorting_list, mem);
		//cuda::memcpy(&mem.num_ptcls, mem.ptclCounter, sizeof(int32_t), cudaMemcpyDeviceToHost);
		//get<parameters::internal::num_ptcls>() = mem.num_ptcls;
		//get<parameters::adaptive::splitPtcls>() = mem.num_ptcls - old;
		int tmpcnt = 0;
		for (int i = 0; i < mem.num_ptcls; i++) {
			if (mem.optimization_group[i].val > 0 && mem.optimization_lifetime[i].val > 0 && mem.particle_type[i] != 0)
			{
				tmpcnt++;
			}
		}
		std::cout << "!!!!!!!! POST SPLITTING " << tmpcnt << std::endl;
	}
	
	cuda::sync();
	auto min = algorithm::reduce_min(mem.volume.first, mem.num_ptcls);
	auto max = PI4O3 * math::power<3>(mem.radius);
	auto ratio = max / min;
	get<parameters::adaptive::minVolume>() = min.val;
	get<parameters::adaptive::ratio>() = ratio.val;
	cuda::sync();
}



float clamp(float val, float mn, float mx)
{
	if (val < mn) val = mn;
	if (val > mx) val = mx;
	return val;
}

neighFunctionType removeExcess(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	return;
	if (arrays.mergeable[i] == -1)
	{
		float probability = (float)(arrays.neighborListLength[i] - arrays.max_neighbors) / arrays.neighborListLength[i] * 100 + 10.f;
		//printf("------------PROBABLE %f %d %d\n", probability, arrays.neighborListLength[i], arrays.max_neighbors);
#if defined(__CUDA_ARCH__)
		curandState state;
		curand_init(1234, 0, 0, &state);
		bool whether_to_delete = curand_uniform(&state) * 100 < probability;
		//whether_to_delete = false;

#else
		bool whether_to_delete = (rand() % 100) < probability;
#endif

		if (whether_to_delete)
		{
			math::unit_assign<4>(arrays.position.first[i], float_u<SI::m>(FLT_MAX));
		}
		return;
	}
}

neighFunctionType postOptimizeMerge(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	return;
	//if (arrays.particle_type[i] == 0) return;
	
	
	auto pos = arrays.position.first;
	if (arrays.particle_type[i] == 0) return;
	if (arrays.optimization_lifetime_merge[i].val <= 2.f) return;
	arrays.optimization_lifetime_merge[i].val = arrays.optimization_lifetime_merge[i].val - 1;
	printf("-------MERGE %d\n", arrays.optimization_lifetime_merge[i].val);
	/*printf("asfsaf00````\n");
	std::vector<float4> vals;
	printf("asfsaf!!!00\n");
	simpleGrid pGrid(vals, pos[i].val.w);*/
	float err = 0;
	float preverr = 10;
	float diferr = 10;
	float diflimit = 0.0001;
	int	  iter = 0;
	int   iterlimit = 100;
	float gradstep = 0.00001;
	float gradsteplim = 0.0001;
	//printf("pos1 %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);

	while (gradstep > gradsteplim && iter < iterlimit)
	{
		int trueResolution = arrays.sdf_resolution + 20;
		float* grd = (float*)arrays.rigidbody_sdf + trueResolution*trueResolution*trueResolution*(arrays.particle_type[i] - 1);
		
		float grdsize = arrays.sdf_gridsize[arrays.particle_type[i] - 1];
		
		float3 currentPoint0 = { pos[i].val.x, pos[i].val.y, pos[i].val.z };
		float signedDistance0 = sdf::lookupSDF(grd, currentPoint0, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]);
		float4 gradient0{
			(sdf::lookupSDF(grd, currentPoint0 + float3{ arrays.sdf_epsilon, 0, 0 }, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance0) / arrays.sdf_epsilon,
			(sdf::lookupSDF(grd, currentPoint0 + float3{ 0, arrays.sdf_epsilon, 0 }, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance0) / arrays.sdf_epsilon,
			(sdf::lookupSDF(grd, currentPoint0 + float3{0, 0, arrays.sdf_epsilon}, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance0) / arrays.sdf_epsilon,
			0
		};
		gradient0 /= math::length3(gradient0);



		float_u<> unit_density = 0.f;
		iterateNeighbors(j) {
			if (arrays.particle_type[j] != 0)
				unit_density += arrays.volume.first[j] * W_ij;
		}
		int sgn = err > 0 ? 1 : -1;

		float4 grad = { 0, 0, 0, 0 };
		iterateNeighbors(j) {
			//if (arrays.particle_type[j] != 0)
				grad += arrays.volume.first[j] * GPW_ij;
		}

		//grad.z = 0.f; ///////////////////remove it after integration of any shapes
		grad = grad - math::dot3(gradient0, grad) / math::dot3(gradient0, gradient0) * gradient0;

		float h = pos[i].val.w;
		pos[i] -= gradstep * grad;
		pos[i].val.w = h;

		unit_density = 0.f;
		float_u<> tmp_density = 0.f;
		int nei = 0;
		iterateNeighbors(j) {
			nei++;
			/*if (arrays.particle_type[j] != 0)
			{*/
				unit_density += arrays.volume.first[j] * W_ij;
				if (i == j) tmp_density += arrays.volume.first[j] * W_ij;
				//printf("index %d indexnei %d ker %f\n", i, j, (arrays.volume.first[j] * W_ij).val);
			//}
		}
		
		err = abs(1 - unit_density.val);
		diferr = abs(err - preverr);
		//if (diferr < diflimit /*|| err > preverr*/ || (iter != 0 && iter % 10 == 0)) {
		//	if (err > preverr) {
		//		float h = pos[i].val.w;
		//		pos[i] += gradstep * grad;
		//		pos[i].val.w = h;
		//	}
		//	gradstep /= 2;
		//	//printf("gradstep %f\n", gradstep);
		//	//printf("pos %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);
		//}
		preverr = err;
		iter++;

		
		auto original_point = float3{ arrays.position.first[i].val.x, arrays.position.first[i].val.y, arrays.position.first[i].val.z };
		//auto original_point = float3{ 1.f, 1.f, 11.5f };

		float dist = sdf::lookupSDF(grd, original_point, grdsize, arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]);
		auto ppoint = sdf::projectOntoMesh(grd, original_point, 1, grdsize, arrays.sdf_resolution, arrays.sdf_epsilon, 10, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]);
		/*printf("initial dist %f \n", dist);
		printf("initial pointXXXXX %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);
		printf("projected pointXXXXX %f %f %f\n", ppoint.x, ppoint.y, ppoint.z);*/

		float3 currentPoint = { pos[i].val.x, pos[i].val.y, pos[i].val.z };
		float signedDistance = sdf::lookupSDF(grd, currentPoint, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]);
		float3 gradient{
			(sdf::lookupSDF(grd, currentPoint + float3{ arrays.sdf_epsilon, 0, 0 }, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance) / arrays.sdf_epsilon,
			(sdf::lookupSDF(grd, currentPoint + float3{ 0, arrays.sdf_epsilon, 0 }, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance) / arrays.sdf_epsilon,
			(sdf::lookupSDF(grd, currentPoint + float3{0, 0, arrays.sdf_epsilon}, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance) / arrays.sdf_epsilon
		};
		gradient /= math::length3(gradient);

		float sum_radius = math::power<ratio<1, 3>>(arrays.volume.first[i] * PI4O3_1).val;
		float rat = (1 - sum_radius / arrays.radius.val);
		
		float elevation = 0.4 * arrays.radius.val * rat;
		//elevation = 0.f;
		gradient *= elevation;
		
		pos[i].val.x = ppoint.x + gradient.x;
		pos[i].val.y = ppoint.y + gradient.y;
		pos[i].val.z = ppoint.z + gradient.z;

		//printf("uid %d iter %d err %f dens %f tmpdens %f smooth %f nei %d vol %f\n",  
		//	arrays.uid[i], iter, err, unit_density.val, tmp_density.val, pos[i].val.w, nei, arrays.volume.first[i].val);
	}		
	//printf("pos2 %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);
	
}


neighFunctionType copyDensity(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	arrays.old_density[i] = arrays.density[i];
}
neighFunction(postoptimizemerge1, postOptimizeMerge, "Adaptive: init (share)");
neighFunction(removeexcess1, removeExcess, "Adaptive: init (share)");
neighFunction(copydensity1, copyDensity, "Adaptive: init (share)");

void SPH::adaptive::copydensity(Memory mem) {
	launch<copydensity1>(mem.num_ptcls, mem);
}

void SPH::adaptive::removeexcess(Memory mem) {
	launch<removeexcess1>(mem.num_ptcls, mem);
}


template<neighbor_list neighborhood>
__global__ void optimizeThread1(SPH::adaptive::Memory arrays, int* cnt, int* sArray, int* indx, int* strtinc) {
	
	//return;
	// the global shared memory state. In total 32 float4 elements and 113 float elements are used for a total of
	// 964 Byte of shared memory per block. With most gpu architectures 64KB of SM are available per SMX which means
	// that the limit of blocks based on SM with a block size of 96 is 67 which is larger than the theoretical HW limit
	// of 21 so no limit to occupancy exists here.
	//DBG_STMNT	printf("%d - [%d : %d] -> %d : %d @ %d\n", __LINE__, blockIdx.x, threadIdx.x, offset, offset + blockIdx.x, ptcls);
	//if (threadIdx.x == 0) printf("TTTTTTTTTTTTTTTTTTTTTTTTTT1 %d %d\n", cnt, strtinc);
	int32_t idx = threadIdx.x;
	int32_t blck = blockIdx.x;
	if (idx > cnt[blck]) {
		//printf("thtreadid %d cnt %d\n", idx, cnt);
		return;
	}
	int idd = idx + strtinc[blck];
	int i = sArray[indx[idd]];
	
	auto cur_radius = math::power<ratio<1, 3>>(arrays.volume.first[i].val * PI4O3_1);
	
	auto pos = arrays.position.first;
	float err = 0;
	float preverr = 10;
	float diferr = 10;
	float diflimit = 0.0001;
	int	  iter = 0;
	int   iterlimit = 100;
	float gradstep = 0.01 * cur_radius;
	float gradsteplim = 0.00001;
	//printf("pos1 %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);
	

	if (arrays.particle_type[i] == 0) return;
	if (arrays.optimization_lifetime[i] <= 0 && arrays.optimization_lifetime_merge[i] <= 0) return;
	arrays.optimization_lifetime[i] = arrays.optimization_lifetime[i] - 1;
	arrays.optimization_lifetime_merge[i] = arrays.optimization_lifetime_merge[i] - 1;
	/*if (arrays.optimization_lifetime_merge[i] > 0) 
		printf("lifetime %d lifetimemerge %d optgroup %d uid %d\n", arrays.optimization_lifetime[i], arrays.optimization_lifetime_merge[i], arrays.optimization_group[i].val, arrays.uid[i]);
	*///printf("TEST2 %d\n", i);
	//auto lala = sArray[0];
	if (arrays.optimization_lifetime[i] <= 0 && arrays.optimization_lifetime_merge[i] <= 0) arrays.optimization_group[i].val = -1;
	//printf("ID %d XYZ %f %f %f \n", i, pos[i].val.x, pos[i].val.y, pos[i].val.z);
	while (gradstep > gradsteplim && iter < iterlimit)
	{
		int trueResolution = arrays.sdf_resolution + 20;
		float* grd = (float*)arrays.rigidbody_sdf + trueResolution*trueResolution*trueResolution*(arrays.particle_type[i] - 1);
		
		float grdsize = arrays.sdf_gridsize[arrays.particle_type[i] - 1];
		
		float3 currentPoint0 = { pos[i].val.x, pos[i].val.y, pos[i].val.z };
		float signedDistance0 = sdf::lookupSDF(grd, currentPoint0, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]);
		float4 gradient0{
			(sdf::lookupSDF(grd, currentPoint0 + float3{ arrays.sdf_epsilon, 0, 0 }, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance0) / arrays.sdf_epsilon,
			(sdf::lookupSDF(grd, currentPoint0 + float3{ 0, arrays.sdf_epsilon, 0 }, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance0) / arrays.sdf_epsilon,
			(sdf::lookupSDF(grd, currentPoint0 + float3{0, 0, arrays.sdf_epsilon}, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance0) / arrays.sdf_epsilon,
			0
		};
		gradient0 /= math::length3(gradient0);

		__syncthreads();
		float_u<> unit_density = 0.f;
		iterateNeighbors(j) {
			if (arrays.particle_type[j] != 0)
				unit_density += arrays.volume.first[j] * W_ij;
		}
		int sgn = err > 0 ? 1 : -1;

		float4 grad = { 0, 0, 0, 0 };
		iterateNeighbors(j) {
			//if (arrays.particle_type[j] != 0)
				grad += arrays.volume.first[j] * GPW_ij;
		}
		int sgn2 = (1 - unit_density) > 0 ? 1 : -1;
		grad = -grad * sgn2;
		grad = grad - math::dot3(gradient0, grad) / math::dot3(gradient0, gradient0) * gradient0;

		//auto grad1 = grad / math::length3(grad);
		/*if (math::length3(grad) != 0.f)
			grad = grad / math::length3(grad);*/
		//grad.z = 0.f; ///////////////////remove it after integration of any shapes
		float h = arrays.position.first[i].val.w;
		
		auto newstep = gradstep * grad;
		
		//printf("newsteplen %f\n", newstep_len);

		arrays.position.first[i] -= newstep;
		arrays.position.first[i].val.w = h;
		

		unit_density = 0.f;
		float_u<> tmp_density = 0.f;
		int nei = 0;
		iterateNeighbors(j) {
			nei++;
			/*if (arrays.particle_type[j] != 0)
			{*/
				unit_density += arrays.volume.first[j] * W_ij;
				if (i == j) tmp_density += arrays.volume.first[j] * W_ij;
				//printf("index %d indexnei %d ker %f\n", i, j, (arrays.volume.first[j] * W_ij).val);
			//}
		}
		
		err = abs(1 - unit_density.val);
		diferr = abs(err - preverr);
		//if (diferr < diflimit /*|| err > preverr*/ || (iter != 0 && iter % 10 == 0)) {
		//	if (err > preverr) {
		//		float h = pos[i].val.w;
		//		pos[i] += gradstep * grad;
		//		pos[i].val.w = h;
		//	}
		//	gradstep /= 2;
		//	printf("gradstep %f\n", gradstep);
		//	//printf("pos %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);
		//}
		preverr = err;
				
		auto original_point = float3{ arrays.position.first[i].val.x, arrays.position.first[i].val.y, arrays.position.first[i].val.z };
		//auto original_point = float3{ 1.f, 1.f, 11.5f };

		float dist = sdf::lookupSDF(grd, original_point, grdsize, arrays.sdf_resolution, arrays.sdf_minpoint);
		auto ppoint = sdf::projectOntoMesh(grd, original_point, cur_radius, grdsize, arrays.sdf_resolution, arrays.sdf_epsilon, 10, arrays.sdf_minpoint, i);
		//printf("initial dist %f \n", dist);
		//printf("ind %d dist %f init p %f %f %f projp %f %f %f\n", i, dist, pos[i].val.x, pos[i].val.y, pos[i].val.z, ppoint.x, ppoint.y, ppoint.z);
		//printf("projected pointXXXXX %f %f %f\n", ppoint.x, ppoint.y, ppoint.z);
		float3 currentPoint = { pos[i].val.x, pos[i].val.y, pos[i].val.z };
		float signedDistance = sdf::lookupSDF(grd, currentPoint, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]);
		float3 gradient{
			(sdf::lookupSDF(grd, currentPoint + float3{ arrays.sdf_epsilon, 0, 0 }, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance) / arrays.sdf_epsilon,
			(sdf::lookupSDF(grd, currentPoint + float3{ 0, arrays.sdf_epsilon, 0 }, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance) / arrays.sdf_epsilon,
			(sdf::lookupSDF(grd, currentPoint + float3{0, 0, arrays.sdf_epsilon}, arrays.sdf_gridsize[arrays.particle_type[i] - 1], arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance) / arrays.sdf_epsilon
		};
		gradient /= math::length3(gradient);

		float sum_radius = math::power<ratio<1, 3>>(arrays.volume.first[i] * PI4O3_1).val;
		float rat = sum_radius / arrays.radius.val;
		
		float elevation = 0.4 * arrays.radius.val * rat;
		//elevation = 0.f;
		gradient *= elevation;
		
		/*if (pos[i].val.x > 10.5 || pos[i].val.x < -10.5)
			printf("XXXXXX %f proj %f grad %f uid %d dens %f grad %f %f %f\n", pos[i].val.x, ppoint.x, gradient.x, arrays.uid[i], unit_density.val, grad.x, grad.y, grad.z);*/

		pos[i].val.x = ppoint.x + gradient.x;
		pos[i].val.y = ppoint.y + gradient.y;
		pos[i].val.z = ppoint.z + gradient.z;

		/*if (pos[i].val.x > 10.5 || pos[i].val.x < -10.5)
			printf("XXXXXX2 %f proj %f grad %f uid %d dens %f grad %f %f %f\n", pos[i].val.x, ppoint.x, gradient.x, arrays.uid[i], unit_density.val, grad.x, grad.y, grad.z);*/

		iter++;
		//printf("uid %d iter %d err %f dens %f tmpdens %f smooth %f nei %d vol %f\n",  
		//	arrays.uid[i], iter, err, unit_density.val, tmp_density.val, pos[i].val.w, nei, arrays.volume.first[i].val);
		__syncthreads();
	}
	//printf("ID %d XYZ %f %f %f \n", i, pos[i].val.x, pos[i].val.y, pos[i].val.z);

	
}

template<neighbor_list neighborhood>
__global__ void optimizeThread(SPH::adaptive::Memory arrays, int* cnt, int* sArray, int* indx, int* strtinc) {
	
	//return;
	// the global shared memory state. In total 32 float4 elements and 113 float elements are used for a total of
	// 964 Byte of shared memory per block. With most gpu architectures 64KB of SM are available per SMX which means
	// that the limit of blocks based on SM with a block size of 96 is 67 which is larger than the theoretical HW limit
	// of 21 so no limit to occupancy exists here.
	//DBG_STMNT	printf("%d - [%d : %d] -> %d : %d @ %d\n", __LINE__, blockIdx.x, threadIdx.x, offset, offset + blockIdx.x, ptcls);
	//if (threadIdx.x == 0) printf("TTTTTTTTTTTTTTTTTTTTTTTTTT1 %d %d\n", cnt, strtinc);
	int32_t idx = threadIdx.x;
	int32_t blck = blockIdx.x;
	if (idx > cnt[blck]) {
		//printf("thtreadid %d cnt %d\n", idx, cnt);
		return;
	}

	
	int idd = idx + strtinc[blck];
	int i = sArray[indx[idd]];
	
	auto cur_radius = math::power<ratio<1, 3>>(arrays.volume.first[i].val * PI4O3_1);
	
	auto pos = arrays.position.first;
	float err = 0;
	float preverr = 10;
	float diferr = 10;
	float diflimit = 0.0001;
	int	  iter = 0;
	int   iterlimit = 100;
	float gradstep = 0.001;
	float gradsteplim = 0.00001;
	//printf("pos1 %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);
	
	if (arrays.particle_type[i] == 0) return;
	if (arrays.optimization_lifetime[i] <= 0 && arrays.optimization_lifetime_merge[i] <= 0) return;
	arrays.optimization_lifetime[i] = arrays.optimization_lifetime[i] - 1;
	arrays.optimization_lifetime_merge[i] = arrays.optimization_lifetime_merge[i] - 1;
	/*if (arrays.optimization_lifetime_merge[i] > 0) 
		printf("lifetime %d lifetimemerge %d optgroup %d uid %d\n", arrays.optimization_lifetime[i], arrays.optimization_lifetime_merge[i], arrays.optimization_group[i].val, arrays.uid[i]);
	*///printf("TEST2 %d\n", i);
	//auto lala = sArray[0];
	int trueResolution = arrays.sdf_resolution + 20;
	float* grd = (float*)arrays.rigidbody_sdf + trueResolution*trueResolution*trueResolution*(arrays.particle_type[i] - 1);
	float grdsize = arrays.sdf_gridsize[arrays.particle_type[i] - 1];
	//float dist1 = sdf::lookupSDF(grd, {0.f, 0.f, 0.f}, grdsize, arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]);
	//printf("initial dist %f partype %d\n", dist1, arrays.particle_type[i]);
		
	if (arrays.optimization_lifetime[i] <= 0 && arrays.optimization_lifetime_merge[i] <= 0) arrays.optimization_group[i].val = -1;
	//printf("ID %d XYZ %f %f %f \n", i, pos[i].val.x, pos[i].val.y, pos[i].val.z);
	while (gradstep > gradsteplim && iter < iterlimit)
	{
		__syncthreads();
		float_u<> unit_density = 0.f;
		iterateNeighbors(j) {
			if (arrays.particle_type[j] != 0)
				unit_density += arrays.volume.first[j] * W_ij;
		}
		int sgn = err > 0 ? 1 : -1;

		float4 grad = { 0, 0, 0, 0 };
		iterateNeighbors(j) {
			if (arrays.particle_type[j] != 0)
				grad += arrays.volume.first[j] * GPW_ij;
		}


		//auto grad1 = grad / math::length3(grad);
		/*if (math::length3(grad) != 0.f)
			grad = grad / math::length3(grad);*/
		//grad.z = 0.f; ///////////////////remove it after integration of any shapes
		//printf("grad %f %f %f norgrad %f %f %f\n", grad.x, grad.y, grad.z, grad1.x, grad1.y, grad1.z);
		float h = pos[i].val.w;
		pos[i] -= gradstep * grad;
		pos[i].val.w = h;
		

		unit_density = 0.f;
		float_u<> tmp_density = 0.f;
		int nei = 0;
		iterateNeighbors(j) {
			nei++;
			if (arrays.particle_type[j] != 0)
			{
				unit_density += arrays.volume.first[j] * W_ij;
				if (i == j) tmp_density += arrays.volume.first[j] * W_ij;
				//printf("index %d indexnei %d ker %f\n", i, j, (arrays.volume.first[j] * W_ij).val);
			}
		}
		
		err = abs(1 - unit_density.val);
		diferr = abs(err - preverr);
		//if (diferr < diflimit /*|| err > preverr*/ || (iter != 0 && iter % 10 == 0)) {
		//	if (err > preverr) {
		//		float h = pos[i].val.w;
		//		pos[i] += gradstep * grad;
		//		pos[i].val.w = h;
		//	}
		//	gradstep /= 2;
		//	printf("gradstep %f\n", gradstep);
		//	//printf("pos %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);
		//}
		preverr = err;
		
		
		auto original_point = float3{ arrays.position.first[i].val.x, arrays.position.first[i].val.y, arrays.position.first[i].val.z };
		//auto original_point = float3{ 1.f, 1.f, 11.5f };

		float dist = sdf::lookupSDF(grd, original_point, grdsize, arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]);
		auto ppoint = sdf::projectOntoMesh(grd, original_point, cur_radius, grdsize, arrays.sdf_resolution, arrays.sdf_epsilon, 10, arrays.sdf_minpoint2[arrays.particle_type[i] - 1], i);
		//printf("initial dist %f \n", dist);
		//printf("ind %d dist %f init p %f %f %f projp %f %f %f\n", i, dist, pos[i].val.x, pos[i].val.y, pos[i].val.z, ppoint.x, ppoint.y, ppoint.z);
		//printf("projected pointXXXXX %f %f %f\n", ppoint.x, ppoint.y, ppoint.z);
		float3 currentPoint = { pos[i].val.x, pos[i].val.y, pos[i].val.z };
		float signedDistance = sdf::lookupSDF(grd, currentPoint, grdsize, arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]);
		float3 gradient{
			(sdf::lookupSDF(grd, currentPoint + float3{ arrays.sdf_epsilon, 0, 0 }, grdsize, arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance) / arrays.sdf_epsilon,
			(sdf::lookupSDF(grd, currentPoint + float3{ 0, arrays.sdf_epsilon, 0 }, grdsize, arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance) / arrays.sdf_epsilon,
			(sdf::lookupSDF(grd, currentPoint + float3{0, 0, arrays.sdf_epsilon}, grdsize, arrays.sdf_resolution, arrays.sdf_minpoint2[arrays.particle_type[i] - 1]) - signedDistance) / arrays.sdf_epsilon
		};
		gradient /= math::length3(gradient);

		float sum_radius = math::power<ratio<1, 3>>(arrays.volume.first[i] * PI4O3_1).val;
		float rat = sum_radius / arrays.radius.val;
		
		float elevation = 0.4 * arrays.radius.val * rat;
		//elevation = 0.f;
		gradient *= elevation;
		
		pos[i].val.x = ppoint.x + gradient.x;
		pos[i].val.y = ppoint.y + gradient.y;
		pos[i].val.z = ppoint.z + gradient.z;

		////printf("ZZZ %f %f %f \n", pos[i].val.z, ppoint.z, gradient.z);

		iter++;
		//printf("uid %d iter %d err %f dens %f tmpdens %f smooth %f nei %d vol %f\n",  
		//	arrays.uid[i], iter, err, unit_density.val, tmp_density.val, pos[i].val.w, nei, arrays.volume.first[i].val);
		__syncthreads();
	}
	//printf("ID %d XYZ %f %f %f \n", i, pos[i].val.x, pos[i].val.y, pos[i].val.z);

	
}


void SPH::adaptive::postoptimizemerge(Memory mem) {
	//return;
	std::cout << "B1\n";
	int32_t stepSize = 200;
	int* idsD;
	int* ids;
	int* groupsD;
	int* groups;
	int* indices;
	int* indicesD;
	int* tmp;
	static int* global_indexD;
	static int* global_index;
	int arrsz = mem.num_ptcls;
	int* cntD, *cnt;
	int* strtindD, *strtind;

	cudaMalloc(&cntD, sizeof(int));
	cudaMalloc(&strtindD, sizeof(int));
	cudaMalloc(&global_indexD, sizeof(int));
	cuda::Memset(global_indexD, -1, sizeof(int));
	global_index = new int;
	cnt = new int;
	strtind = new int;
	*global_index = 0;
	cudaMalloc(&idsD, sizeof(int) * arrsz);
	cuda::Memset(idsD, 0, sizeof(int) * arrsz);
	ids = new int[arrsz];
	cudaMalloc(&groupsD, sizeof(int) * arrsz);
	cuda::Memset(groupsD, -1, sizeof(int) * arrsz);
	groups = new int[arrsz];
	cudaMalloc(&indicesD, sizeof(int) * arrsz);
	indices = new int[arrsz];
	for (int i = 0; i < arrsz; i++)
	{
		indices[i] = i;
		ids[i] = 0;
	}

	std::cout << "B3\n";
	std::vector<int> group_ids;
	std::vector<int> group_amounts;
	std::vector<int> particle_ind;
	std::vector<int> particle_group;
	for (int i = 0; i < mem.num_ptcls; i++) {
		if (mem.optimization_lifetime_merge[i].val > 0 && mem.particle_type[i] != 0)
		{
			particle_ind.push_back(i);
			particle_group.push_back(mem.optimization_group[i].val);
		}
		/*countForOpt<neighbor_list::constrained> <<<stepSize, 96 >>> (mem, i * stepSize * 96, mem.num_ptcls, get<parameters::internal::frame>() + i, idsD, groupsD, global_index);
		cuda::sync();
	*/}

	//cuda::sync();
	//for (int i = 0; i < global_index; i++)
	

	
	if (particle_ind.size() > 0)
	{
		auto tmparr = particle_group.data();
		std::sort(indices, indices+particle_ind.size(), sort_indices(tmparr));
		pairsort(tmparr, indices, particle_group.size());
		std::cout << "B4.1\n";
		//for (int j = 0; j < particle_ind.size(); j++) {
		//	std::cout << " index " << indices[j] << " group " << particle_group[j] << std::endl;
		//}
		
		int* tmpind = particle_ind.data();
		cudaMemcpy(idsD, tmpind, sizeof(int) * particle_ind.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(indicesD, indices, sizeof(int) * arrsz, cudaMemcpyHostToDevice);
		

		*strtind = 0;
		int endind = 0;
		//int cnt = 0;
		int curid = -1;
		//std::cout << "gloablind " << *global_indexH << "\n";
		int tobeoptimizedFlag = 0;
		std::cout << "Before OPTIMIZE " << particle_ind.size() << "\n";
		int numofgroups = 0;
		for (int i = 0; i < particle_ind.size(); i++)
		{
			//std::cout << " break0 \n";

			if (curid == -1) {
				curid = particle_group[indices[i]];
				*strtind = i;
			}
			
			//std::cout << "groupid " << particle_group[indices[i]] << "\n";
			if (curid != particle_group[indices[i]] || particle_group.size() == i + 1) {
				*cnt = i - *strtind;
				group_amounts.push_back(*cnt);
				group_ids.push_back(*strtind);
				numofgroups++;
		
				curid = particle_group[indices[i]];
				*strtind = i;
				if (particle_group[indices[i]] == -1) break;
			}
			
			//std::cout << " i END " << "\n";


			
		}
		std::cout << "!!!!!!!!!!!!!!!!!!!!!!!MERGE OPT " << numofgroups << " \n";

		int* group_amountsD;
		cudaMalloc(&group_amountsD, sizeof(int) * numofgroups);
		cudaMemcpy(group_amountsD, &group_amounts[0], sizeof(int) * numofgroups, cudaMemcpyHostToDevice);
		
		int* group_idsD;
		cudaMalloc(&group_idsD, sizeof(int) * numofgroups);
		cudaMemcpy(group_idsD, &group_ids[0], sizeof(int) * numofgroups, cudaMemcpyHostToDevice);

		cuda::sync();
		optimizeThread1<neighbor_list::constrained> <<<numofgroups, 256 >>> (mem, group_amountsD, idsD, indicesD, group_idsD);
		cuda::sync();

				std::cout << "After OPTIMIZE\n";
		
	}
	std::cout << "B5!!!!!!!!!!!!!!!!!!!!!!!!\n";
	free(cnt);
	free(strtind);
	free(global_index);
	free(ids);
	free(groups);
	free(indices);
	cudaFree(cntD);
	cudaFree(strtindD);
	cudaFree(global_indexD);
	cudaFree(idsD);
	cudaFree(groupsD);
	cudaFree(indicesD);
}






void SPH::adaptive::postcontoptimizeCount(Memory mem) {
	//possible bug. here i use only one type of neighbor list neighbor_list::constrained. it might cause problems with other types

	std::cout << "B1\n";
	int32_t stepSize = 200;
	int* idsD;
	int* ids;
	int* groupsD;
	int* groups;
	int* indices;
	int* indicesD;
	int* tmp;
	static int* global_indexD;
	static int* global_index;
	int arrsz = mem.num_ptcls;
	int* cntD, *cnt;
	int* strtindD, *strtind;

	cudaMalloc(&cntD, sizeof(int));
	cudaMalloc(&strtindD, sizeof(int));
	cudaMalloc(&global_indexD, sizeof(int));
	cuda::Memset(global_indexD, -1, sizeof(int));
	global_index = new int;
	cnt = new int;
	strtind = new int;
	*global_index = 0;
	cudaMalloc(&idsD, sizeof(int) * arrsz);
	cuda::Memset(idsD, 0, sizeof(int) * arrsz);
	ids = new int[arrsz];
	cudaMalloc(&groupsD, sizeof(int) * arrsz);
	cuda::Memset(groupsD, -1, sizeof(int) * arrsz);
	groups = new int[arrsz];
	cudaMalloc(&indicesD, sizeof(int) * arrsz);
	indices = new int[arrsz];
	for (int i = 0; i < arrsz; i++)
	{
		indices[i] = i;
		ids[i] = 0;
	}

	std::cout << "B3\n";
	std::vector<int> group_ids;
	std::vector<int> group_amounts;
	std::vector<int> particle_ind;
	std::vector<int> particle_group;
	for (int i = 0; i < mem.num_ptcls; i++) {
		if (mem.optimization_group[i].val > 0 && mem.optimization_lifetime[i].val > 0 && mem.particle_type[i] != 0)
		{
			particle_ind.push_back(i);
			particle_group.push_back(mem.optimization_group[i].val);
			mem.tobeoptimized[i] = -1;
		}
		/*countForOpt<neighbor_list::constrained> <<<stepSize, 96 >>> (mem, i * stepSize * 96, mem.num_ptcls, get<parameters::internal::frame>() + i, idsD, groupsD, global_index);
		cuda::sync();
	*/}

	//cuda::sync();
	//for (int i = 0; i < global_index; i++)
	

	
	if (particle_ind.size() > 0)
	{
		auto tmparr = particle_group.data();
		std::sort(indices, indices+particle_ind.size(), sort_indices(tmparr));
		pairsort(tmparr, indices, particle_group.size());
		std::cout << "B4.1\n";
		//for (int j = 0; j < particle_ind.size(); j++) {
		//	std::cout << " index " << indices[j] << " group " << particle_group[j] << std::endl;
		//}
		
		int* tmpind = particle_ind.data();
		cudaMemcpy(idsD, tmpind, sizeof(int) * particle_ind.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(indicesD, indices, sizeof(int) * arrsz, cudaMemcpyHostToDevice);
		

		*strtind = 0;
		int endind = 0;
		//int cnt = 0;
		int curid = -1;
		//std::cout << "gloablind " << *global_indexH << "\n";
		int tobeoptimizedFlag = 0;
		bool fflag = false;
		std::cout << "Before OPTIMIZE " << particle_ind.size() << "\n";
		int numofgroups = 0;
		for (int i = 0; i < particle_ind.size(); i++)
		{
			if (particle_group[indices[i]] > -1 && ids[indices[i]] > -1 && mem.tobeoptimized[ids[indices[i]]] > 0)
				fflag = true;
			//std::cout << " break0 \n";

			if (curid == -1) {
				curid = particle_group[indices[i]];
				*strtind = i;
			}
			
			//std::cout << "groupid " << particle_group[indices[i]] << "\n";
			if (curid != particle_group[indices[i]] || particle_group.size() == i + 1) {
				*cnt = i - *strtind;
				group_amounts.push_back(*cnt);
				group_ids.push_back(*strtind);
				numofgroups++;
		
				curid = particle_group[indices[i]];
				*strtind = i;
				if (particle_group[indices[i]] == -1) break;
			}
			
			//std::cout << " i END " << "\n";


			
		}

		int* group_amountsD;
		cudaMalloc(&group_amountsD, sizeof(int) * numofgroups);
		cudaMemcpy(group_amountsD, &group_amounts[0], sizeof(int) * numofgroups, cudaMemcpyHostToDevice);
		
		int* group_idsD;
		cudaMalloc(&group_idsD, sizeof(int) * numofgroups);
		cudaMemcpy(group_idsD, &group_ids[0], sizeof(int) * numofgroups, cudaMemcpyHostToDevice);

		cuda::sync();
		optimizeThread<neighbor_list::constrained> <<<numofgroups, 256 >>> (mem, group_amountsD, idsD, indicesD, group_idsD);
		cuda::sync();

				std::cout << "After OPTIMIZE\n";
		
	}
	std::cout << "B5!!!!!!!!!!!!!!!!!!!!!!!!\n";
	free(cnt);
	free(strtind);
	free(global_index);
	free(ids);
	free(groups);
	free(indices);
	cudaFree(cntD);
	cudaFree(strtindD);
	cudaFree(global_indexD);
	cudaFree(idsD);
	cudaFree(groupsD);
	cudaFree(indicesD);
}

void SPH::adaptive::calculateVol(Memory mem) {
	//std::cout << "+++++FLUIDS: " << algorithm::count_if(arrays::particle_type::ptr, get<parameters::internal::num_ptcls>(), is_fluid()) << " " << get<parameters::internal::num_ptcls>() << "\n";
	//std::cout << "+++++REFINED: " << algorithm::count_if(arrays::optimization_group::ptr, get<parameters::internal::num_ptcls>(), is_refined()) << " " << get<parameters::internal::num_ptcls>() << "\n";
	/*std::cout << "-------------------------------------------------calculatefluid----------------------------------\n";*/
	float totalarea = 0.f;
	for (int i = 0; i < mem.num_ptcls; i++)
	{
		if (mem.particle_type[i] == 0) continue;
		float area = M_PI * math::pow(3 * mem.volume.first[i].val / 4 / M_PI, 2.0 / 3.0);
		totalarea += area;
		//std::cout << "*********AREA " << area << " uid " << mem.uid[i] << " h " << mem.position.first[i].val.w << std::endl;
	}
	std::cout << "*********TOTAL AREA " << totalarea << std::endl;
}