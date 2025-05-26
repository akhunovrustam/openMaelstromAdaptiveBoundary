// As this module uses openVDB which can conflict with the custom math operators used for cuda we
// disable them here.
#define NEW_STYLE
//#define NO_OPERATORS
#define BOOST_USE_WINDOWS_H
#include <SPH/generation/volumeOutlets.cuh>
#include <utility/include_all.h>
#ifdef _WIN32
#pragma warning(push, 0)
#endif
#include <openvdb/Platform.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tree/ValueAccessor.h>
#ifdef _WIN32
#pragma warning(pop)
#endif
// At some point this will have to be replaced with <filesystem>
#include <fstream>
#ifdef _WIN32
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif

#include <utility/generation.h>


template <typename Tag, typename Tag::type M> struct Robber {
    friend typename Tag::type get(Tag) { return M; }
};

// tag used to access A::member
struct As_member {
    typedef CUdeviceptr VolumeGVDB::* type;
    friend type get(As_member);
};

template struct Robber<As_member, &VolumeGVDB::cuVDBInfo>;

std::vector<VolumeGVDB> gvdbVolumeManagers2;


std::pair<float4, float4> getGVDBTransform2(std::string fileName) {
    typedef openvdb::tree::Tree<openvdb::tree::RootNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::LeafNode<float, 4>, 3>, 3>, 3>>> FloatTree34;
    typedef openvdb::tree::Tree<openvdb::tree::RootNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::LeafNode<openvdb::Vec3f, 4>, 3>, 3>, 3>>> Vec3fTree34;
    typedef openvdb::Grid<FloatTree34>		FloatGrid34;
    typedef openvdb::Grid<Vec3fTree34>		Vec3fGrid34;
    typedef FloatGrid34						GridType34;
    typedef FloatGrid34::TreeType			TreeType34F;
    typedef Vec3fGrid34::TreeType			TreeType34VF;
    typedef openvdb::FloatGrid				FloatGrid543;
    typedef openvdb::Vec3fGrid				Vec3fGrid543;
    typedef FloatGrid543					GridType543;
    typedef FloatGrid543::TreeType			TreeType543F;
    typedef Vec3fGrid543::TreeType			TreeType543VF;

    FloatGrid543::Ptr			grid543F;
    Vec3fGrid543::Ptr			grid543VF;
    FloatGrid34::Ptr			grid34F;
    Vec3fGrid34::Ptr			grid34VF;
    // iterators
    TreeType543F::LeafCIter		iter543F;
    TreeType543VF::LeafCIter	iter543VF;
    TreeType34F::LeafCIter		iter34F;
    TreeType34VF::LeafCIter		iter34VF;


    auto vdbSkip = [&](int leaf_start, int gt, bool isFloat)
    {
        switch (gt) {
        case 0:
            if (isFloat) { iter543F = grid543F->tree().cbeginLeaf();  for (int j = 0; iter543F && j < leaf_start; j++) ++iter543F; }
            else { iter543VF = grid543VF->tree().cbeginLeaf(); for (int j = 0; iter543VF && j < leaf_start; j++) ++iter543VF; }
            break;
        case 1:
            if (isFloat) { iter34F = grid34F->tree().cbeginLeaf();  for (int j = 0; iter34F && j < leaf_start; j++) ++iter34F; }
            else { iter34VF = grid34VF->tree().cbeginLeaf(); for (int j = 0; iter34VF && j < leaf_start; j++) ++iter34VF; }
            break;
        };
    };
    auto vdbCheck = [&](int gt, bool isFloat)
    {
        switch (gt) {
        case 0: return (isFloat ? iter543F.test() : iter543VF.test());	break;
        case 1: return (isFloat ? iter34F.test() : iter34VF.test());	break;
        };
        return false;
    };
    auto vdbOrigin = [&](openvdb::Coord& orig, int gt, bool isFloat)
    {
        switch (gt) {
        case 0: if (isFloat) (iter543F)->getOrigin(orig); else (iter543VF)->getOrigin(orig);	break;
        case 1: if (isFloat) (iter34F)->getOrigin(orig); else (iter34VF)->getOrigin(orig);	break;
        };
    };
    auto vdbNext = [&](int gt, bool isFloat)
    {
        switch (gt) {
        case 0: if (isFloat) iter543F.next();	else iter543VF.next();		break;
        case 1: if (isFloat) iter34F.next();	else iter34VF.next();		break;
        };
    };


    openvdb::CoordBBox box;
    openvdb::Coord orig;
    Vector3DF p0, p1;

    PERF_PUSH("Clear grid");

    PERF_POP();

    PERF_PUSH("Load VDB");

    // Read .vdb file	

    openvdb::io::File* vdbfile = new openvdb::io::File(fileName);
    vdbfile->open();

    // Read grid		
    openvdb::GridBase::Ptr baseGrid;
    openvdb::io::File::NameIterator nameIter = vdbfile->beginName();
    std::string name = vdbfile->beginName().gridName();
    baseGrid = vdbfile->readGrid(name);
    PERF_POP();

    // Initialize GVDB config
    Vector3DF voxelsize;
    int gridtype = 0;

    bool isFloat = false;

    if (baseGrid->isType< FloatGrid543 >()) {
        gridtype = 0;
        isFloat = true;
        grid543F = openvdb::gridPtrCast< FloatGrid543 >(baseGrid);
        voxelsize.Set(grid543F->voxelSize().x(), grid543F->voxelSize().y(), grid543F->voxelSize().z());
    }
    if (baseGrid->isType< Vec3fGrid543 >()) {
        gridtype = 0;
        isFloat = false;
        grid543VF = openvdb::gridPtrCast< Vec3fGrid543 >(baseGrid);
        voxelsize.Set(grid543VF->voxelSize().x(), grid543VF->voxelSize().y(), grid543VF->voxelSize().z());
    }
    if (baseGrid->isType< FloatGrid34 >()) {
        gridtype = 1;
        isFloat = true;
        grid34F = openvdb::gridPtrCast< FloatGrid34 >(baseGrid);
        voxelsize.Set(grid34F->voxelSize().x(), grid34F->voxelSize().y(), grid34F->voxelSize().z());
    }
    if (baseGrid->isType< Vec3fGrid34 >()) {
        gridtype = 1;
        isFloat = false;
        grid34VF = openvdb::gridPtrCast< Vec3fGrid34 >(baseGrid);
        voxelsize.Set(grid34VF->voxelSize().x(), grid34VF->voxelSize().y(), grid34VF->voxelSize().z());
    }

    slong leaf;
    int leaf_start = 0;				// starting leaf		gScene.mVLeaf.x;		
    int n, leaf_max, leaf_cnt = 0;
    Vector3DF vclipmin{ -FLT_MAX, -FLT_MAX, -FLT_MAX }, vclipmax{ FLT_MAX, FLT_MAX, FLT_MAX }, voffset;
    Vector3DF mVoxMin, mVoxMax;
    // Determine Volume bounds
    vdbSkip(leaf_start, gridtype, isFloat);
    for (leaf_max = 0; vdbCheck(gridtype, isFloat); ) {
        vdbOrigin(orig, gridtype, isFloat);
        p0.Set(orig.x(), orig.y(), orig.z());
        if (p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z) {		// accept condition
            if (leaf_max == 0) {
                mVoxMin = p0; mVoxMax = p0;
            }
            else {
                if (p0.x < mVoxMin.x) mVoxMin.x = p0.x;
                if (p0.y < mVoxMin.y) mVoxMin.y = p0.y;
                if (p0.z < mVoxMin.z) mVoxMin.z = p0.z;
                if (p0.x > mVoxMax.x) mVoxMax.x = p0.x;
                if (p0.y > mVoxMax.y) mVoxMax.y = p0.y;
                if (p0.z > mVoxMax.z) mVoxMax.z = p0.z;
            }
            leaf_max++;
        }
        vdbNext(gridtype, isFloat);
    }
    voffset = mVoxMin * -1;		// offset to positive space (hack)	

    return std::make_pair(float4{ voffset.x, voffset.y, voffset.z ,0.f }, 1.f / float4{ voxelsize.x, voxelsize.y, voxelsize.z,1.f });
}

// This function is used to load the vdb files from disk and transforms them into cuda 3d textures.
void SPH::Outlet::init(Memory mem) {
  if (!valid(mem)) {
    return;
  }

  //static VolumeGVDB gvdbInstance2;
  auto vols = get<parameters::outlet_volumes::volume>();

  std::vector<cudaTextureObject_t> textures;
  std::vector<float4> minAABB;
  std::vector<float4> maxAABB;
  std::vector<int4> dims;
  std::vector<float> rates;
  std::vector<VDBInfo*> gvdbVolumes;
  std::vector<float4> gvdbOffsets;
  std::vector<float4> gvdbVoxelSizes;

  for (auto boundaryVolume : vols) {
	auto[texture, min, max, dimension, centerOfMass, inertia] = generation::cudaVolume(boundaryVolume.fileName);
    // Load VDB
    auto [gptr, offset, size] = gvdbHelper::loadIntoGVDB(boundaryVolume.fileName);

    gvdbVolumes.push_back(gptr);
    gvdbOffsets.push_back(offset/* + centerOfMass * size*/);
    gvdbVoxelSizes.push_back(size);

    //auto ming = gvdbInstance2.getWorldMin();
    //auto maxg = gvdbInstance2.getWorldMax();
    //std::cout << "ming: " << ming.x << " " << ming.y << " " << ming.z << std::endl;
    //std::cout << "maxg: " << maxg.x << " " << maxg.y << " " << maxg.z << std::endl;

    //std::cout << "minv: " << minVDB.x() << " " << minVDB.y() << " " << minVDB.z() << std::endl;
    //std::cout << "maxv: " << maxVDB.x() << " " << maxVDB.y() << " " << maxVDB.z() << std::endl;

    //std::cout << "minb: " << box.getStart().x() << " " << box.getStart().y() << " " << box.getStart().z() << std::endl;
    //std::cout << "maxb: " << box.getEnd().x() << " " << box.getEnd().y() << " " << box.getEnd().z() << std::endl;


	min += centerOfMass;
	max += centerOfMass;
    //std::cout << "min: " << min.x << " " << min.y << " " << min.z << std::endl;
    //std::cout << "max: " << max.x << " " << max.y << " " << max.z << std::endl;
    textures.push_back(texture);
    minAABB.push_back(float4{static_cast<float>(min.x), static_cast<float>(min.y), static_cast<float>(min.z), 0.f});
    maxAABB.push_back(float4{static_cast<float>(max.x), static_cast<float>(max.y), static_cast<float>(max.z), 0.f});
    dims.push_back(dimension);
    rates.push_back(boundaryVolume.flowRate);
    //free(host_mem);
  }
  if (textures.size() > 0) {
    using dimensions = decltype(info<arrays::volumeOutletDimensions>());
    using boundaryMins = decltype(info<arrays::volumeOutletMin>());
    using boundaryMaxs = decltype(info<arrays::volumeOutletMax>());
    using rate = decltype(info<arrays::volumeOutletRate>());
    using rateAcc = decltype(info<arrays::volumeOutletRateAccumulator>());
    using gvdbvols = decltype(info<arrays::outletGVDBVolumes>());
    using Awtransforms = decltype(info<arrays::volumeOutletOffsets>());
    using AwinverseTransforms = decltype(info<arrays::volumeOutletVoxelSizes>());

    get<parameters::outlet_volumes::volumeOutletCounter>() = (int32_t) textures.size();

    dimensions::allocate(sizeof(int4) * textures.size());
    gvdbvols::allocate(sizeof(gvdbvols::type) * textures.size());
    boundaryMins::allocate(sizeof(float4) * textures.size());
    boundaryMaxs::allocate(sizeof(float4) * textures.size());
    rate::allocate(sizeof(float) * textures.size());
    rateAcc::allocate(sizeof(float) * textures.size());
    Awtransforms::allocate(sizeof(float4) * textures.size());
    AwinverseTransforms::allocate(sizeof(float4) * textures.size());

#ifdef UNIFIED_MEMORY
	for (uint32_t i = 0; i < textures.size(); ++i) {
		dimensions::ptr[i] = dims[i];
		volumes::ptr[i] = textures[i];
		boundaryMins::ptr[i] = minAABB[i];
		boundaryMaxs::ptr[i] = maxAABB[i];
        rate::ptr[i] = rates[i];
        rateAcc::ptr[i] = 0.f;
  }
#else
	cudaMemcpy(dimensions::ptr, dims.data(), sizeof(decltype(dims)::value_type) * dims.size(), cudaMemcpyHostToDevice);
	//cudaMemcpy(volumes::ptr, textures.data(), sizeof(decltype(textures)::value_type) * textures.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(boundaryMins::ptr, minAABB.data(), sizeof(decltype(minAABB)::value_type) * minAABB.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(boundaryMaxs::ptr, maxAABB.data(), sizeof(decltype(maxAABB)::value_type) * maxAABB.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(rate::ptr, rates.data(), sizeof(decltype(rates)::value_type) * rates.size(), cudaMemcpyHostToDevice);
	cudaMemset(rateAcc::ptr,0x00,sizeof(float) * textures.size());

    cuda::memcpy(gvdbvols::ptr, gvdbVolumes.data(), sizeof(decltype(gvdbVolumes)::value_type) * gvdbVolumes.size(),
        cudaMemcpyHostToDevice);
    cuda::memcpy(Awtransforms::ptr, gvdbOffsets.data(), sizeof(Awtransforms::type) * textures.size(),
        cudaMemcpyHostToDevice);
    cuda::memcpy(AwinverseTransforms::ptr, gvdbVoxelSizes.data(), sizeof(AwinverseTransforms::type) * textures.size(),
        cudaMemcpyHostToDevice);
#endif
  }
  else {
	  using dimensions = decltype(info<arrays::volumeOutletDimensions>());
	  using volumes = decltype(info<arrays::outletGVDBVolumes>());
	  using boundaryMins = decltype(info<arrays::volumeOutletMin>());
	  using boundaryMaxs = decltype(info<arrays::volumeOutletMax>());
      using rate = decltype(info<arrays::volumeOutletRate>());
      using rateAcc = decltype(info<arrays::volumeOutletRateAccumulator>());

	  get<parameters::outlet_volumes::volumeOutletCounter>() = 0;

	  dimensions::allocate(sizeof(int4) * 1);
	  volumes::allocate(sizeof(cudaTextureObject_t) * 1);
	  boundaryMins::allocate(sizeof(float4) * 1);
	  boundaryMaxs::allocate(sizeof(float4) * 1);
      rate::allocate(sizeof(float) * 1);
      rateAcc::allocate(sizeof(float) * 1);
  }
}