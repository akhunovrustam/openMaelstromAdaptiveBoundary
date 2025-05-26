#pragma once
#define NOMINMAX
#include <GLFW/glfw3.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <fstream>


struct simpleGrid {
	double h = 1.0;
	int32_t cx, cy, cz;
	int32_t n;
	openvdb::Vec4f min = openvdb::Vec4f(FLT_MAX, FLT_MAX, FLT_MAX, 0);
	openvdb::Vec4f max = openvdb::Vec4f(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0);
	std::vector<std::vector<int32_t>> cells;
	
	auto getCell(const openvdb::Vec4f& v) {
		auto sx = (v.x() - min.x()) / h ;
		auto sy = (v.y() - min.y()) / h ;
		auto sz = (v.z() - min.z()) / h ;
		auto rx = sx * (float) cx;
		auto ry = sy * (float) cy;
		auto rz = sz * (float) cz;
		auto ix = std::clamp(static_cast<int32_t>(std::floor(sx)), -1, cx );
		auto iy = std::clamp(static_cast<int32_t>(std::floor(sy)), -1, cy );
		auto iz = std::clamp(static_cast<int32_t>(std::floor(sz)), -1, cz );
		// std::clog << v << " => [ " << sx << " x " << sy << " x " << sz << " ], [ " << rx << " x " << ry << " x " << rz << " ], [ " << ix << " x " << iy << " x " << iz << " ] " << std::endl;
		return std::make_tuple(ix + 1, iy + 1, iz + 1);
	}
	auto getLinear(const openvdb::Vec4f& v) {
		auto [x, y, z] = getCell(v);
		// std::clog << "linear " << x + (cx + 2) * (y + (cy + 2) * z) << std::endl;
		return x + (cx + 3) * (y + (cy + 3) * z);
	}
	simpleGrid(const std::vector<openvdb::Vec4f>& values, double _h) :h(_h) {
		for (const auto& v : values) {
			min.x() = std::min(min.x(), v.x());
			min.y() = std::min(min.y(), v.y());
			min.z() = std::min(min.z(), v.z());
			max.x() = std::max(max.x(), v.x());
			max.y() = std::max(max.y(), v.y());
			max.z() = std::max(max.z(), v.z());
		}
		min -= 0.01;
		cx = (max.x() - min.x()) / h;
		cy = (max.y() - min.y()) / h;
		cz = (max.z() - min.z()) / h;
		n = (cx + 3) * (cy + 3) * (cz + 3) + 1;
		cells.resize(n);
		// std::clog << "Cell information: " << std::endl;
		// std::clog << "domain: " << min << " -> " << max << ", h = " << h << std::endl;
		// std::clog << "cells: [ " << cx << " x " << cy << " x " << cz << " ] => " << n << std::endl;
		// std::clog << "particleCount: " << values.size() << std::endl;
		for (int32_t i = 0; i < values.size(); ++i)
			cells[getLinear(values[i])].push_back(i);
	}
	auto reInitialize(const std::vector<openvdb::Vec4f>& values) {
	min = openvdb::Vec4f(FLT_MAX, FLT_MAX, FLT_MAX, 0);
	max = openvdb::Vec4f(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0);
		for (const auto& v : values) {
			min.x() = std::min(min.x(), v.x());
			min.y() = std::min(min.y(), v.y());
			min.z() = std::min(min.z(), v.z());
			max.x() = std::max(max.x(), v.x());
			max.y() = std::max(max.y(), v.y());
			max.z() = std::max(max.z(), v.z());
		}
		min -= 0.01;
		cx = (max.x() - min.x()) / h;
		cy = (max.y() - min.y()) / h;
		cz = (max.z() - min.z()) / h;
		auto nn = (cx + 3) * (cy + 3) * (cz + 3) + 1;
#pragma omp parallel for
		for (int32_t i = 0; i < n; ++i)
			cells[i].clear();
		if (nn != n)
			cells.resize(nn);
		n = nn;
		for (int32_t i = 0; i < values.size(); ++i)
			cells[getLinear(values[i])].push_back(i);
	}

	template<typename C>
	auto iterate(const openvdb::Vec4f& v, C&& func) {
		auto [x, y, z] = getCell(v);
		//std::clog << "x = " << v << " => [ " << x << " x " << y << " x " << z << " ]" << std::endl;
		for (int32_t ix = -1; ix <= 1; ++ix) {
			for (int32_t iy = -1; iy <= 1; ++iy) {
				for (int32_t iz = -1; iz <= 1; ++iz) {
					auto linear = x + ix + (cx + 3) * (y + iy + (cy + 3) * (z + iz));
					//std::clog << "c = [ " << ix << " x " << iy << " x " << iz << " ] => " << linear << std::endl;
					//std::clog << "cellSize = " << cells[linear].size() << std::endl;
 					for (auto i : cells[linear])
						func(i);
				}
			}
		}
	}
};


std::vector<openvdb::Vec4f> particles;
std::vector<openvdb::Vec4f> sampleOptimized(std::string file) {
	std::clog << "Sampling object from file " << file << std::endl;
	// sampling configuration parameters
	const float voxelSize = 0.25f;
	const float particleSize = 0.5f / 1.0f;
	//const float particleSize = 0.5f / 3.17f;
	auto density = 3.5f;
	auto impulseIterations = 300;
	auto impulseScale = 0.01;

	auto h = particleSize * 2.0;

	struct Triangle {
		int32_t i0, i1, i2;
	};
	struct Edge {
		int32_t start, end;
	};
	namespace vdb = openvdb;
	namespace vdbt = openvdb::tools;
	namespace vdbm = openvdb::math;
	constexpr auto PI = 3.14159265358979323846;

	vdb::Vec3d min_vdb, max_vdb;
	vdb::FloatGrid::Ptr grid;
	int32_t tris;

	
	std::vector<vdb::Vec4f> vertices;
	std::vector<vdb::Vec4f> normals;
	std::vector<int32_t> normal_counter;
	std::vector<Triangle> triangles;
	std::vector<Edge> edges;

	std::vector<vdb::Vec4f> particles;

	std::string warn, err;
    // CommonFileIOInterface* fileIO;

	// auto errr = tinyobj::LoadObj(shapes, file.c_str(), "./Configurations//DamBreakWithRigids//Volumes/", fileIO);
    auto [objVertices, objPlanes, objEdges, objMin, objMax] = generation::fileToObj(file.c_str());
	
	std::clog << "Loaded obj file via tinyobjloader" << std::endl;
	std::vector<vdb::Vec3s> vdbVertices;
	std::vector<vdb::Vec3I> vdbIndices;
	std::vector<vdb::Vec4I> vdbIndices2;
    for(auto v : objVertices){
        vdbVertices.push_back(vdb::Vec3s(v.x, v.y, v.z));
    }
    for(auto t : objPlanes){
        vdbIndices.push_back(vdb::Vec3I(t.i0, t.i1, t.i2));
    }


	// for (int32_t i = 0; i < tIdxs.size() / 3; ++i) {
	// 	auto i0 = tIdxs[i * 3 + 0] * 3;
	// 	auto i1 = tIdxs[i * 3 + 1] * 3;
	// 	auto i2 = tIdxs[i * 3 + 2] * 3;
	// 	vdbVertices.push_back(vdb::Vec3s(tVtxs[i0], tVtxs[i0 + 1], tVtxs[i0 + 2]));
	// 	vdbVertices.push_back(vdb::Vec3s(tVtxs[i1], tVtxs[i1 + 1], tVtxs[i1 + 2]));
	// 	vdbVertices.push_back(vdb::Vec3s(tVtxs[i2], tVtxs[i2 + 1], tVtxs[i2 + 2]));
	// 	vdbIndices.push_back(vdb::Vec3I(i * 3, i * 3 + 1, i * 3 + 2));
	// }
	std::clog << "Transformed data into openVDB compatible format" << std::endl;
	vdbm::Transform::Ptr xform = vdbm::Transform::createLinearTransform(voxelSize);
	grid = vdbt::meshToSignedDistanceField<vdb::FloatGrid>(*xform, vdbVertices, vdbIndices, vdbIndices2, 25.f, 5.f);
	vdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
	min_vdb = grid->indexToWorld(box.getStart());
	max_vdb = grid->indexToWorld(box.getEnd());
	std::clog << "Converted mesh into signed distance field" << std::endl;

	auto getNormal = [](auto e0, auto e1, auto e2) {
		auto e1_e0 = e1 - e0;
		openvdb::Vec3d e11 = openvdb::Vec3d{e1_e0.x(), e1_e0.y(), e1_e0.z()};
		auto e2_e0 = e2 - e0;
		openvdb::Vec3d e22 = openvdb::Vec3d{e2_e0.x(), e2_e0.y(), e2_e0.z()};
		auto n = e11.cross(e22);
		auto l = n.length();
		float ttt = n.x() / l;
		if (l > 1e-7f)
			return vdb::Vec4f{ ttt, (float)(n.y() / l), (float)(n.z() / l), 0 };
		else
			return vdb::Vec4f{ 0.f, 1.f, 0.f, 0 };
	};
	openvdb::Vec3d min{ (float)min_vdb.x(), (float)min_vdb.y(), (float)min_vdb.z() };
	openvdb::Vec3d max{ (float)max_vdb.x(), (float)max_vdb.y(), (float)max_vdb.z() };
	vdbt::VolumeToMesh mesher(0.0, 0.0);
	mesher(*grid);
	std::clog << "Extracted iso surface from signed distance field" << std::endl;
	for (vdb::Index64 n = 0, i = 0, N = mesher.pointListSize(); n < N; ++n) {
		const vdb::Vec3s& p = mesher.pointList()[n];
		vertices.push_back(vdb::Vec4f{ p.x(), p.y(), p.z(), 0 });
		normals.push_back(vdb::Vec4f{ 0.f, 0.f, 0.f, 0.f });
		normal_counter.push_back(0);
	}
	vdbt::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();
	for (vdb::Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
		const vdbt::PolygonPool& polygons = polygonPoolList[n];
		for (vdb::Index64 i = 0, I = polygons.numQuads(); i < I; ++i) {
			const vdb::Vec4I& quad = polygons.quad(i);
			auto i0 = (int32_t)quad[0];
			auto i1 = (int32_t)quad[1];
			auto i2 = (int32_t)quad[2];
			auto i3 = (int32_t)quad[3];
			triangles.push_back(Triangle{ i0, i1, i2 });
			triangles.push_back(Triangle{ i3, i2, i0 });
			edges.push_back(Edge{ i0, i1 });
			edges.push_back(Edge{ i1, i2 });
			edges.push_back(Edge{ i2, i3 });
			edges.push_back(Edge{ i3, i0 });
			edges.push_back(Edge{ i0, i2 });
			auto e0 = vertices[i0];
			auto e1 = vertices[i1];
			auto e2 = vertices[i2];
			auto e3 = vertices[i3];

			auto n1 = getNormal(e0, e1, e2);
			auto n2 = getNormal(e2, e3, e0);
			normals[i0] += n1 + n2;
			normal_counter[i0] += 2;
			normals[i1] += n1;
			normal_counter[i1] += 1;
			normals[i2] += n1 + n2;
			normal_counter[i2] += 2;
			normals[i3] += n2;
			normal_counter[i3] += 1;
		}
		for (vdb::Index64 i = 0, I = polygons.numTriangles(); i < I; ++i) {
			const vdb::Vec3I& quad = polygons.triangle(i);
			auto i0 = (int32_t)quad[0];
			auto i1 = (int32_t)quad[1];
			auto i2 = (int32_t)quad[2];
			triangles.push_back(Triangle{ i0, i1, i2 });
			edges.push_back(Edge{ i0, i1 });
			edges.push_back(Edge{ i1, i2 });
			edges.push_back(Edge{ i2, i0 });
			auto e0 = vertices[i0];
			auto e1 = vertices[i1];
			auto e2 = vertices[i2];

			auto n1 = getNormal(e0, e1, e2);
			normals[i0] += n1;
			normal_counter[i0]++;
			normals[i1] += n1;
			normal_counter[i1]++;
			normals[i2] += n1;
			normal_counter[i2]++;
		}
	}
	for (int32_t i = 0; i < normals.size(); ++i)
		normals[i] = (normal_counter[i] == 0 ? vdb::Vec4f{ 0.f, 1.f, 0.f, 0 } : normals[i] / (float)normal_counter[i]);

	std::vector<int32_t> indices;
	for (const auto& t : triangles) {
		indices.push_back(t.i0);
		indices.push_back(t.i1);
		indices.push_back(t.i2);
	}
	tris = (int32_t)indices.size();
	std::clog << "Transformed vdb mesh representation into useful format" << std::endl;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-10.0, 10.0);
	std::uniform_real_distribution<> dis1(0.0, 1.0);

	auto a = PI * particleSize * particleSize;
	for (auto t : triangles) {
		auto p0 = vertices[t.i0];
		auto p1 = vertices[t.i1];
		auto p2 = vertices[t.i2];

		auto at = (p0 - p1).length();
		auto bt = (p1 - p2).length();
		auto ct = (p2 - p0).length();
		auto p = at + bt + ct;
		auto s = p * 0.5;
		auto area = density * std::sqrt(s * (s - at) * (s - bt) * (s - ct));
		auto ptcls = static_cast<int32_t>(area / a) + (dis1(gen) < area / a - std::floor(area / a) ? 1 : 0);
		for (int32_t i = 0; i < ptcls; ++i) {
			auto s = dis1(gen);
			auto t = dis1(gen);
			if (s + t > 1.0) {
				s = 1.0 - s;
				t = 1.0 - t;
			}
			auto p = p0 + t * (p1 - p0) + s * (p2 - p0);
			particles.push_back(openvdb::Vec4f{(float)p.x(), (float)p.y(), (float)p.z(), 0.f});
		}
	}
	std::clog << "Finished random sampling of boundary surface with " << particles.size() << " particles" << std::endl;

	vdbt::GridSampler<vdb::FloatGrid, vdbt::BoxSampler> d_sampler(*grid);
	vdb::FloatGrid::ConstAccessor c_accessor = grid->getConstAccessor();
	vdbm::UniformScaleMap map(1.0);

	vdb::FloatGrid::ConstAccessor naccessor = grid->getConstAccessor();
	vdbt::GridSampler<vdb::FloatGrid::ConstAccessor, vdbt::BoxSampler> normalSampler(naccessor, grid->transform());

	
	std::clog << "Created openvdb grid for normal lookup" << std::endl;

	std::vector<std::vector<int32_t>> particleGrid;
	auto dvdb = (max_vdb - min_vdb) / h;
	int32_t cx = std::ceil(dvdb.x());
	int32_t cy = std::ceil(dvdb.y());
	int32_t cz = std::ceil(dvdb.z());
	int32_t hx = cx + 2;
	int32_t hy = cy + 2;
	int32_t hz = cz + 2;
	particleGrid.resize((hx) * (hy) * (hz));


	auto evalLS = [&](vdb::Vec4f pp) {
		auto dx = 0.05;
		openvdb::Vec3d p = openvdb::Vec3d{pp.x(), pp.y(), pp.z()};
		auto phi = d_sampler.wsSample(p);
		auto phix = d_sampler.wsSample(p + vdb::Vec3d(1.0, 0.0, 0.0) * dx);
		auto phiy = d_sampler.wsSample(p + vdb::Vec3d(0.0, 1.0, 0.0) * dx);
		auto phiz = d_sampler.wsSample(p + vdb::Vec3d(0.0, 0.0, 1.0) * dx);
		return std::make_pair(phi, (vdb::Vec3d(phix, phiy, phiz) - phi) / dx);
	};
	
	auto getIdx = [&](vdb::Vec3d p) {
		auto rel = (p - min_vdb) / h;
		auto ix = std::clamp(static_cast<int32_t>(std::floor(rel.x())), 0, cx) + 1;
		auto iy = std::clamp(static_cast<int32_t>(std::floor(rel.y())), 0, cy) + 1;
		auto iz = std::clamp(static_cast<int32_t>(std::floor(rel.z())), 0, cz) + 1;
		return std::make_tuple(ix, iy, iz);
	};
	simpleGrid pGrid(particles, h);
	std::clog << "Started impulse optimization" << std::endl;
// 	for (int32_t x = 0; x < impulseIterations; ++x) {
// 		std::cout << "\tImpulse iteration " << x << std::endl;
// 		std::vector<vdb::Vec4f> newPositions(particles.begin(), particles.end());
// 		pGrid.reInitialize(particles);
// 		std::clog << "\tParticle Count " << particles.size() << std::endl;
// #pragma omp parallel for
// 		for (int32_t i = 0; i < particles.size(); ++i) {
// 			auto& p = particles[i];
// 			auto [phi, n] = evalLS(p);
// 			//auto [phi, n] = evalVDB(p);
// 			auto vff = -phi * n;
// 			vdb::Vec4f vf = vdb::Vec4f{vff.x(), vff.y(), vff.z(), 0};
// 			auto vr = vdb::Vec4f(0, 0, 0, 0);
// 			pGrid.iterate(p, [&](int32_t j) {
// 				//std::clog << j << " / " << particles.size() << std::endl;
// 				auto& pj = particles[j];
// 				auto diff = p - pj;
// 				auto d = diff.length();
// 				auto q = d / h;
// 				if (d < h && d > 1e-5f) {
// 					auto dir = diff.normalize();
// 					auto q1 = 1.0 - q;
// 					auto q2 = 0.5 - q;
// 					vr += diff * ((q1 * q1 * q1) + (q <= 0.5 ? -4.0 * q2 * q2 * q2 : 0.0));
// 				}});
// 			auto normgrad = n / (n.length() > 1e-5 ? n.length() : 1);
// 			vr.normalize();
// 			vr *= impulseScale * 10.0;			
// 			newPositions[i] = p + vf + vr;
// 		}
// 		std::clog << "\tStep done" << std::endl;
// 		particles = newPositions;
// 	}
	for (int32_t x = 0; x < impulseIterations; ++x) {
		std::cout << "\tImpulse iteration " << x << std::endl;
		std::vector<vdb::Vec4f> newPositions(particles.begin(), particles.end());
		pGrid.reInitialize(particles);
		std::clog << "\tParticle Count " << particles.size() << std::endl;
#pragma omp parallel for
		for (int32_t i = 0; i < particles.size(); ++i) {
			auto& p = particles[i];
			auto [phi, n] = evalLS(p);
			//auto [phi, n] = evalVDB(p);
			auto vff = -phi * n;
			vdb::Vec4f vf = vdb::Vec4f{(float)vff.x(), (float)vff.y(), (float)vff.z(), 0};
			auto vr = vdb::Vec4f(0, 0, 0, 0);
			pGrid.iterate(p, [&](int32_t j) {
				//std::clog << j << " / " << particles.size() << std::endl;
				auto& pj = particles[j];
				auto diff = p - pj;
				auto d = diff.length();
				auto q = d / h;
				if (d < h && d > 1e-5f) {
					auto dir = diff.normalize();
					auto q1 = 1.0 - q;
					auto q2 = 0.5 - q;
					vr += diff * ((q1 * q1 * q1) + (q <= 0.5 ? -4.0 * q2 * q2 * q2 : 0.0));
				}});
			auto normgrad = n / (n.length() > 1e-5 ? n.length() : 1);
			vr.normalize();
			vr *= PI * particleSize * particleSize * impulseScale / (h * h) * 80.f / (7.f * PI);
			newPositions[i] = p + vf + vr;
		}
		std::clog << "\tStep done" << std::endl;
		particles = newPositions;
	}
	std::clog << "Pushing particles onto SDF" << std::endl;
#pragma omp parallel for
	for (int32_t i = 0; i < particles.size(); ++i) {
		auto& p = particles[i];
		auto [phi, n] = evalLS(p);
		auto vff = -phi * n;
		vdb::Vec4f vf = vdb::Vec4f{(float)vff.x(), (float)vff.y(), (float)vff.z(), 0};
		particles[i] += vf;
	}
	std::clog << "\tDone with particle generation" << std::endl;
	return particles;
}

std::vector<openvdb::Vec4f> sampleOptimizedDens(std::string file) {
	std::clog << "Sampling object from file " << file << std::endl;
	// sampling configuration parameters
	const float voxelSize = 0.25f / 2.9f;
	const float particleSize = 0.5f / 2.9f;
	auto density = 2.0f;
	auto impulseIterations = 30;
	auto impulseScale = 0.1;

	auto h = particleSize * 2.0;

	struct Triangle {
		int32_t i0, i1, i2;
	};
	struct Edge {
		int32_t start, end;
	};
	namespace vdb = openvdb;
	namespace vdbt = openvdb::tools;
	namespace vdbm = openvdb::math;
	constexpr auto PI = 3.14159265358979323846;

	vdb::Vec3d min_vdb, max_vdb;
	vdb::FloatGrid::Ptr grid;
	int32_t tris;

	
	std::vector<vdb::Vec4f> vertices;
	std::vector<vdb::Vec4f> normals;
	std::vector<int32_t> normal_counter;
	std::vector<Triangle> triangles;
	std::vector<Edge> edges;

	std::vector<vdb::Vec4f> particles;

	std::string warn, err;
    // CommonFileIOInterface* fileIO;

	// auto errr = tinyobj::LoadObj(shapes, file.c_str(), "./Configurations//DamBreakWithRigids//Volumes/", fileIO);
    auto [objVertices, objPlanes, objEdges, objMin, objMax] = generation::fileToObj(file.c_str());
	
	std::clog << "Loaded obj file via tinyobjloader" << std::endl;
	std::vector<vdb::Vec3s> vdbVertices;
	std::vector<vdb::Vec3I> vdbIndices;
	std::vector<vdb::Vec4I> vdbIndices2;
    for(auto v : objVertices){
        vdbVertices.push_back(vdb::Vec3s(v.x, v.y, v.z));
    }
    for(auto t : objPlanes){
        vdbIndices.push_back(vdb::Vec3I(t.i0, t.i1, t.i2));
    }


	// for (int32_t i = 0; i < tIdxs.size() / 3; ++i) {
	// 	auto i0 = tIdxs[i * 3 + 0] * 3;
	// 	auto i1 = tIdxs[i * 3 + 1] * 3;
	// 	auto i2 = tIdxs[i * 3 + 2] * 3;
	// 	vdbVertices.push_back(vdb::Vec3s(tVtxs[i0], tVtxs[i0 + 1], tVtxs[i0 + 2]));
	// 	vdbVertices.push_back(vdb::Vec3s(tVtxs[i1], tVtxs[i1 + 1], tVtxs[i1 + 2]));
	// 	vdbVertices.push_back(vdb::Vec3s(tVtxs[i2], tVtxs[i2 + 1], tVtxs[i2 + 2]));
	// 	vdbIndices.push_back(vdb::Vec3I(i * 3, i * 3 + 1, i * 3 + 2));
	// }
	std::clog << "Transformed data into openVDB compatible format" << std::endl;
	vdbm::Transform::Ptr xform = vdbm::Transform::createLinearTransform(voxelSize);
	grid = vdbt::meshToSignedDistanceField<vdb::FloatGrid>(*xform, vdbVertices, vdbIndices, vdbIndices2, 25.f, 5.f);
	vdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
	min_vdb = grid->indexToWorld(box.getStart());
	max_vdb = grid->indexToWorld(box.getEnd());
	std::clog << "Converted mesh into signed distance field" << std::endl;

	auto getNormal = [](auto e0, auto e1, auto e2) {
		auto e1_e0 = e1 - e0;
		openvdb::Vec3d e11 = openvdb::Vec3d{e1_e0.x(), e1_e0.y(), e1_e0.z()};
		auto e2_e0 = e2 - e0;
		openvdb::Vec3d e22 = openvdb::Vec3d{e2_e0.x(), e2_e0.y(), e2_e0.z()};
		auto n = e11.cross(e22);
		auto l = n.length();
		float ttt = n.x() / l;
		if (l > 1e-7f)
			return vdb::Vec4f{ ttt, (float)(n.y() / l), (float)(n.z() / l), 0 };
		else
			return vdb::Vec4f{ 0.f, 1.f, 0.f, 0 };
	};
	openvdb::Vec3d min{ (float)min_vdb.x(), (float)min_vdb.y(), (float)min_vdb.z() };
	openvdb::Vec3d max{ (float)max_vdb.x(), (float)max_vdb.y(), (float)max_vdb.z() };
	vdbt::VolumeToMesh mesher(0.0, 0.0);
	mesher(*grid);
	std::clog << "Extracted iso surface from signed distance field" << std::endl;
	for (vdb::Index64 n = 0, i = 0, N = mesher.pointListSize(); n < N; ++n) {
		const vdb::Vec3s& p = mesher.pointList()[n];
		vertices.push_back(vdb::Vec4f{ p.x(), p.y(), p.z(), 0 });
		normals.push_back(vdb::Vec4f{ 0.f, 0.f, 0.f, 0.f });
		normal_counter.push_back(0);
	}
	vdbt::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();
	for (vdb::Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
		const vdbt::PolygonPool& polygons = polygonPoolList[n];
		for (vdb::Index64 i = 0, I = polygons.numQuads(); i < I; ++i) {
			const vdb::Vec4I& quad = polygons.quad(i);
			auto i0 = (int32_t)quad[0];
			auto i1 = (int32_t)quad[1];
			auto i2 = (int32_t)quad[2];
			auto i3 = (int32_t)quad[3];
			triangles.push_back(Triangle{ i0, i1, i2 });
			triangles.push_back(Triangle{ i3, i2, i0 });
			edges.push_back(Edge{ i0, i1 });
			edges.push_back(Edge{ i1, i2 });
			edges.push_back(Edge{ i2, i3 });
			edges.push_back(Edge{ i3, i0 });
			edges.push_back(Edge{ i0, i2 });
			auto e0 = vertices[i0];
			auto e1 = vertices[i1];
			auto e2 = vertices[i2];
			auto e3 = vertices[i3];

			auto n1 = getNormal(e0, e1, e2);
			auto n2 = getNormal(e2, e3, e0);
			normals[i0] += n1 + n2;
			normal_counter[i0] += 2;
			normals[i1] += n1;
			normal_counter[i1] += 1;
			normals[i2] += n1 + n2;
			normal_counter[i2] += 2;
			normals[i3] += n2;
			normal_counter[i3] += 1;
		}
		for (vdb::Index64 i = 0, I = polygons.numTriangles(); i < I; ++i) {
			const vdb::Vec3I& quad = polygons.triangle(i);
			auto i0 = (int32_t)quad[0];
			auto i1 = (int32_t)quad[1];
			auto i2 = (int32_t)quad[2];
			triangles.push_back(Triangle{ i0, i1, i2 });
			edges.push_back(Edge{ i0, i1 });
			edges.push_back(Edge{ i1, i2 });
			edges.push_back(Edge{ i2, i0 });
			auto e0 = vertices[i0];
			auto e1 = vertices[i1];
			auto e2 = vertices[i2];

			auto n1 = getNormal(e0, e1, e2);
			normals[i0] += n1;
			normal_counter[i0]++;
			normals[i1] += n1;
			normal_counter[i1]++;
			normals[i2] += n1;
			normal_counter[i2]++;
		}
	}
	for (int32_t i = 0; i < normals.size(); ++i)
		normals[i] = (normal_counter[i] == 0 ? vdb::Vec4f{ 0.f, 1.f, 0.f, 0 } : normals[i] / (float)normal_counter[i]);

	std::vector<int32_t> indices;
	for (const auto& t : triangles) {
		indices.push_back(t.i0);
		indices.push_back(t.i1);
		indices.push_back(t.i2);
	}
	tris = (int32_t)indices.size();
	std::clog << "Transformed vdb mesh representation into useful format" << std::endl;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-10.0, 10.0);
	std::uniform_real_distribution<> dis1(0.0, 1.0);

	auto a = PI * particleSize * particleSize;
	for (auto t : triangles) {
		auto p0 = vertices[t.i0];
		auto p1 = vertices[t.i1];
		auto p2 = vertices[t.i2];

		auto at = (p0 - p1).length();
		auto bt = (p1 - p2).length();
		auto ct = (p2 - p0).length();
		auto p = at + bt + ct;
		auto s = p * 0.5;
		auto area = density * std::sqrt(s * (s - at) * (s - bt) * (s - ct));
		auto ptcls = static_cast<int32_t>(area / a) + (dis1(gen) < area / a - std::floor(area / a) ? 1 : 0);
		for (int32_t i = 0; i < ptcls; ++i) {
			auto s = dis1(gen);
			auto t = dis1(gen);
			if (s + t > 1.0) {
				s = 1.0 - s;
				t = 1.0 - t;
			}
			auto p = p0 + t * (p1 - p0) + s * (p2 - p0);
			particles.push_back(openvdb::Vec4f{(float)p.x(), (float)p.y(), (float)p.z(), 0.f});
		}
	}
	std::clog << "Finished random sampling of boundary surface with " << particles.size() << " particles" << std::endl;

	vdbt::GridSampler<vdb::FloatGrid, vdbt::BoxSampler> d_sampler(*grid);
	vdb::FloatGrid::ConstAccessor c_accessor = grid->getConstAccessor();
	vdbm::UniformScaleMap map(1.0);

	vdb::FloatGrid::ConstAccessor naccessor = grid->getConstAccessor();
	vdbt::GridSampler<vdb::FloatGrid::ConstAccessor, vdbt::BoxSampler> normalSampler(naccessor, grid->transform());

	
	std::clog << "Created openvdb grid for normal lookup" << std::endl;

	std::vector<std::vector<int32_t>> particleGrid;
	auto dvdb = (max_vdb - min_vdb) / h;
	int32_t cx = std::ceil(dvdb.x());
	int32_t cy = std::ceil(dvdb.y());
	int32_t cz = std::ceil(dvdb.z());
	int32_t hx = cx + 2;
	int32_t hy = cy + 2;
	int32_t hz = cz + 2;
	particleGrid.resize((hx) * (hy) * (hz));


	auto evalLS = [&](vdb::Vec4f pp) {
		auto dx = 0.05;
		openvdb::Vec3d p = openvdb::Vec3d{pp.x(), pp.y(), pp.z()};
		auto phi = d_sampler.wsSample(p);
		auto phix = d_sampler.wsSample(p + vdb::Vec3d(1.0, 0.0, 0.0) * dx);
		auto phiy = d_sampler.wsSample(p + vdb::Vec3d(0.0, 1.0, 0.0) * dx);
		auto phiz = d_sampler.wsSample(p + vdb::Vec3d(0.0, 0.0, 1.0) * dx);
		return std::make_pair(phi, (vdb::Vec3d(phix, phiy, phiz) - phi) / dx);
	};
	
	auto getIdx = [&](vdb::Vec3d p) {
		auto rel = (p - min_vdb) / h;
		auto ix = std::clamp(static_cast<int32_t>(std::floor(rel.x())), 0, cx) + 1;
		auto iy = std::clamp(static_cast<int32_t>(std::floor(rel.y())), 0, cy) + 1;
		auto iz = std::clamp(static_cast<int32_t>(std::floor(rel.z())), 0, cz) + 1;
		return std::make_tuple(ix, iy, iz);
	};
	simpleGrid pGrid(particles, h);
	std::clog << "Started impulse optimization" << std::endl;
// 	for (int32_t x = 0; x < impulseIterations; ++x) {
// 		std::cout << "\tImpulse iteration " << x << std::endl;
// 		std::vector<vdb::Vec4f> newPositions(particles.begin(), particles.end());
// 		pGrid.reInitialize(particles);
// 		std::clog << "\tParticle Count " << particles.size() << std::endl;
// #pragma omp parallel for
// 		for (int32_t i = 0; i < particles.size(); ++i) {
// 			auto& p = particles[i];
// 			auto [phi, n] = evalLS(p);
// 			//auto [phi, n] = evalVDB(p);
// 			auto vff = -phi * n;
// 			vdb::Vec4f vf = vdb::Vec4f{vff.x(), vff.y(), vff.z(), 0};
// 			auto vr = vdb::Vec4f(0, 0, 0, 0);
// 			pGrid.iterate(p, [&](int32_t j) {
// 				//std::clog << j << " / " << particles.size() << std::endl;
// 				auto& pj = particles[j];
// 				auto diff = p - pj;
// 				auto d = diff.length();
// 				auto q = d / h;
// 				if (d < h && d > 1e-5f) {
// 					auto dir = diff.normalize();
// 					auto q1 = 1.0 - q;
// 					auto q2 = 0.5 - q;
// 					vr += diff * ((q1 * q1 * q1) + (q <= 0.5 ? -4.0 * q2 * q2 * q2 : 0.0));
// 				}});
// 			auto normgrad = n / (n.length() > 1e-5 ? n.length() : 1);
// 			vr.normalize();
// 			vr *= impulseScale * 10.0;			
// 			newPositions[i] = p + vf + vr;
// 		}
// 		std::clog << "\tStep done" << std::endl;
// 		particles = newPositions;
// 	}
	for (int32_t x = 0; x < impulseIterations; ++x) {
		std::cout << "\tImpulse iteration " << x << std::endl;
		std::vector<vdb::Vec4f> newPositions(particles.begin(), particles.end());
		pGrid.reInitialize(particles);
		std::clog << "\tParticle Count " << particles.size() << std::endl;
		auto vol = PI4O3 * particleSize;
//#pragma omp parallel for
		for (int32_t i = 0; i < particles.size(); ++i) {
			auto& p = particles[i];
			auto [phi, n] = evalLS(p);
			//auto [phi, n] = evalVDB(p);
			auto vff = -phi * n;
			vdb::Vec4f vf = vdb::Vec4f{(float)vff.x(), (float)vff.y(), (float)vff.z(), 0};
			auto vr = vdb::Vec4f(0, 0, 0, 0);
			float_u<> unit_density = 0.f;
			float4 pos1 = float4{ p.x(), p.y(), p.z(), (float)h };
			//pGrid.iterate(p, [&](int32_t j) {
			for (auto pj : particles) {
				//std::clog << j << " / " << particles.size() << std::endl;
				//auto& pj = particles[j];
				float4 pos2 = float4{ pj.x(), pj.y(), pj.z(), (float)h };
				//float4 vrtmp = vol * spline4_kernel(pos1, pos2);
				float4 vrtmp = (vol * PressureKernel<kernel_kind::spline4>::gradient(pos1, pos2));
				vr += vdb::Vec4f(vrtmp.x, vrtmp.y, vrtmp.z, 0);
			}
				//});
			auto normgrad = n / (n.length() > 1e-5 ? n.length() : 1);
			//vr.normalize();
			vr = vr * impulseScale;
			//vr = vdb::Vec4f(1, 0, 0, 0);
			//vr *= PI * particleSize * particleSize * impulseScale / (h * h) * 80.f / (7.f * PI);
			newPositions[i] = p /*+ vf*/ - vr;
			
			auto p1 = vdb::Vec4f{(float)newPositions[i].x(), (float)newPositions[i].y(), (float)newPositions[i].z(), 0};
			auto [phi1, n1] = evalLS(p1);
			auto vff1 = -phi1 * n1;
			vdb::Vec4f vf1 = vdb::Vec4f{(float)vff1.x(), (float)vff1.y(), (float)vff1.z(), 0};
			newPositions[i] = newPositions[i] + vf1;

			newPositions[i].w() = h;
		}
		std::clog << "\tStep done" << std::endl;
		particles = newPositions;
	}
	std::clog << "Pushing particles onto SDF SMOOTH " << particles[0].w() << std::endl;
#pragma omp parallel for
	for (int32_t i = 0; i < particles.size(); ++i) {
		auto& p = particles[i];
		auto [phi, n] = evalLS(p);
		auto vff = -phi * n;
		vdb::Vec4f vf = vdb::Vec4f{(float)vff.x(), (float)vff.y(), (float)vff.z(), 0};
		particles[i] += vf;
	}
	std::clog << "\tDone with particle generation SMOOOTH " << particles[0].w() << std::endl;
	return particles;
}

std::vector<openvdb::Vec4f> densityOptimize(float vol, std::vector<openvdb::Vec4f> pos) {
	
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
	float gradstep = 0.5;
	float gradsteplim = 0.0001;
	//printf("pos1 %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);

	float maxx = -100.f;
	float maxy = -100.f;
	float maxz = -100.f;
	float minx = 100.f;
	float miny = 100.f;
	float minz = 100.f;
	
	for (auto par : pos)
	{
		if (par.x() > maxx) maxx = par.x();
		if (par.x() < minx) minx = par.x();
		if (par.y() > maxy) maxy = par.y();
		if (par.y() < miny) miny = par.y();
		if (par.z() > maxz) maxz = par.z();
		if (par.z() < minz) minz = par.z();
	}

	while (gradstep > gradsteplim && iter < iterlimit)
	{
		std::vector<openvdb::Vec4f> newpos;
		float totaldens = 0.f;
		float sumsqrd = 0.f;
		for (int i = 0; i < pos.size(); i++) {

			float dens = 0.f;
			float4 poss1 = {pos[i].x(), pos[i].y(), pos[i].z(), pos[i].w()};
			float_u<> unit_density = 0.f;
			for (int j = 0; j < pos.size(); j++) {
				float4 poss2 = {pos[j].x(), pos[j].y(), pos[j].z(), pos[j].w()};
				unit_density += vol * spline4_kernel(poss1, poss2);
				dens = spline4_kernel(poss1, poss2);
			}
			int sgn = err > 0 ? 1 : -1;
			
			//std::cout << "dens " << unit_density.val << " " << poss1.x << " " << poss1.y << " " << poss1.z << " " << poss1.w << " " << std::endl;
			totaldens += abs(1 - unit_density.val);
			sumsqrd += abs(1 - unit_density.val) * abs(1 - unit_density.val);

			float4 grad = { 0, 0, 0, 0 };
			for (int j = 0; j < pos.size(); j++) {
				float4 poss2 = {pos[j].x(), pos[j].y(), pos[j].z(), pos[j].w()};
				grad += vol * PressureKernel<kernel_kind::spline4>::gradient(poss1, poss2);
			}
			openvdb::Vec4f tmp;
			auto tmp2 = gradstep * grad;
			tmp.x() = tmp2.x;
			tmp.y() = tmp2.y;
			tmp.z() = tmp2.z;
			openvdb::Vec4f tmpnewpos = pos[i] - tmp;
			tmpnewpos.w() = poss1.w;

			if (tmpnewpos.x() > maxx)
				tmpnewpos.x() = maxx;
			if (tmpnewpos.x() < minx)
				tmpnewpos.x() = minx;
			if (tmpnewpos.y() > maxy)
				tmpnewpos.y() = maxy;
			if (tmpnewpos.y() < miny)
				tmpnewpos.y() = miny;
			if (tmpnewpos.z() > maxz)
				tmpnewpos.z() = maxz;
			if (tmpnewpos.z() < minz)
				tmpnewpos.z() = minz;
			newpos.push_back(tmpnewpos);

			unit_density = 0.f;
			float_u<> tmp_density = 0.f;
			int nei = 0;
			//for (int j = 0; j < pos.size(); j++) {
			//	float4 poss2 = {pos[j].x(), pos[j].y(), pos[j].z(), pos[j].w()};
			//	nei++;
			//	unit_density += vol * spline4_kernel(poss1, poss2);
			//	if (i == j) tmp_density += vol * spline4_kernel(poss1, poss2);
			//	//printf("index %d indexnei %d ker %f\n", i, j, (arrays.volume.first[j] * W_ij).val);
			//	
			//}

			//err += abs(1 - unit_density.val);
			//diferr = abs(err - preverr);
			//if (diferr < diflimit /*|| err > preverr*/ || (iter != 0 && iter % 10 == 0)) {
			//	if (err > preverr) {
			//		openvdb::Vec4f tmp;
			//		auto tmp2 = gradstep * grad;
			//		tmp.x() = tmp2.x;
			//		tmp.y() = tmp2.y;
			//		tmp.z() = tmp2.z;
			//		pos[i] += tmp;
			//		pos[i].w() = poss1.w;
			//	}
			//	gradstep /= 2;
			//	//printf("gradstep %f\n", gradstep);
			//	//printf("pos %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);
			//}
			preverr = err;
			//iter++;
			//printf("uid %d iter %d err %f dens %f tmpdens %f smooth %f nei %d vol %f\n",  
			//	arrays.uid[i], iter, err, unit_density.val, tmp_density.val, pos[i].val.w, nei, arrays.volume.first[i].val);
		}
		pos = newpos;
		printf("iter %d\n", iter);
		iter++;
		auto mean = totaldens / pos.size();
		std::ofstream myfile;
		myfile.open ("errors.txt", std::ios_base::app);
		myfile << mean << " " << sqrt(sumsqrd / pos.size() - mean * mean) << "\n";
		myfile.close();

		
	}		
	//printf("pos2 %f %f %f\n", pos[i].val.x, pos[i].val.y, pos[i].val.z);
	return pos;
}

std::vector<openvdb::Vec4f> impulseOptimize(float vol, std::vector<openvdb::Vec4f> particles) {
	// sampling configuration parameters
	const float particleSize = 0.5f;
	auto impulseIterations = 100;
	auto impulseScale = 0.50;

	auto h = particles[0].w() * 3.0;

	struct Triangle {
		int32_t i0, i1, i2;
	};
	struct Edge {
		int32_t start, end;
	};

	constexpr auto PI = 3.14159265358979323846;

	float maxx = -10.f;
	float maxy = -10.f;
	float maxz = -10.f;
	float minx = 10.f;
	float miny = 10.f;
	float minz = 10.f;
	
	for (auto par : particles)
	{
		if (par.x() > maxx) maxx = par.x();
		if (par.x() < minx) minx = par.x();
		if (par.y() > maxy) maxy = par.y();
		if (par.y() < miny) miny = par.y();
		if (par.z() > maxz) maxz = par.z();
		if (par.z() < minz) minz = par.z();
	}

	std::string warn, err;
    // CommonFileIOInterface* fileIO;

	//simpleGrid pGrid(particles, h);
	for (int32_t x = 0; x < impulseIterations; ++x) {
		float totaldens = 0.f;
		float sumsqrd = 0.f;
		std::cout << "\tImpulse iteration " << x << std::endl;
		std::vector<openvdb::Vec4f> newPositions(particles.begin(), particles.end());
		//pGrid.reInitialize(particles);
		std::clog << "\tParticle Count " << particles.size() << std::endl;
//#pragma omp parallel for
		for (int32_t i = 0; i < particles.size(); ++i) {
			auto& p = particles[i];
			auto vr = openvdb::Vec4f(0, 0, 0, 0);
			//pGrid.iterate(p, [&](int32_t j) {
			//	//std::clog << j << " / " << particles.size() << std::endl;
			//	auto& pj = particles[j];
			//	auto diff = p - pj;
			//	auto d = diff.length();
			//	auto q = d / h;
			//	if (d < h && d > 1e-5f) {
			//		auto dir = diff.normalize();
			//		auto q1 = 1.0 - q;
			//		auto q2 = 0.5 - q;
			//		vr += diff * ((q1 * q1 * q1) + (q <= 0.5 ? -4.0 * q2 * q2 * q2 : 0.0));
			//	}});
			float4 p1 = { p.x(), p.y(), p.z(), p.w() }; 
			float dens = vol * spline4_kernel(p1, p1);
			for (int j = 0; j < particles.size(); j++) {
				if (j == i) continue;
				auto& pj = particles[j];
				float4 p2 = { pj.x(), pj.y(), pj.z(), pj.w() };
				auto diff = p1 - p2;
				auto kr = spline4_kernel(p1, p2);
				dens += vol * kr;
				auto dist = math::distance3(p1, p2);
				auto tmpvr = diff * kr / dist;
				vr += openvdb::Vec4f(tmpvr.x, tmpvr.y, tmpvr.z, 0.f);
				/*std::cout << p1.x << " " << p1.y << " " << p1.z << " " << p1.w << " " << p2.x << " " << p2.y << " " << p2.z << " " << p2.w << "\n";
				std::cout << kr << " " << dist << " " << diff.x << " " << diff.y << " " << diff.z << " " << diff.w << "\n";
				*///std::cout << tmpvr.x << " " << tmpvr.y << " " << tmpvr.z << " " << tmpvr.w << "\n";
				/*auto diff = p - pj;
				auto d = diff.length();
				auto q = d / h;
				if (d < h && d > 1e-5f) {
					auto dir = diff.normalize();
					auto q1 = 1.0 - q;
					auto q2 = 0.5 - q;
					vr += diff * ((q1 * q1 * q1) + (q <= 0.5 ? -4.0 * q2 * q2 * q2 : 0.0));
				}*/
			}
			//std::cout << "clown " << dens << p1.x << " " << p1.y << " " << p1.z << " " << p1.w << " " << std::endl;
			/*std::cout << "seperator\n";
			std::cout << "wft " << dens << " " << p1.x << " " << p1.y << " " << p1.z << " " << p1.w << "\n";
			*/totaldens += abs(1 - dens);
			sumsqrd += abs(1 - dens) * abs(1 - dens);
			
			vr *= impulseScale;
			//std::cout << p.x() << " " << p.y() << " " << p.z() << " " << vr.x() << " " << vr.y() << " " << vr.z() << "\n";
			newPositions[i] = p + vr;
			float thrd = 10.0;
			if (newPositions[i].x() > maxx)
				newPositions[i].x() = maxx;
			if (newPositions[i].x() < minx)
				newPositions[i].x() = minx;
			if (newPositions[i].y() > maxy)
				newPositions[i].y() = maxy;
			if (newPositions[i].y() < miny)
				newPositions[i].y() = miny;
			if (newPositions[i].z() > maxz)
				newPositions[i].z() = maxz;
			if (newPositions[i].z() < minz)
				newPositions[i].z() = minz;
		}
		std::clog << "\tStep done" << std::endl;
		particles = newPositions;
		auto mean = totaldens / particles.size();
		std::ofstream myfile;
		myfile.open ("errors.txt", std::ios_base::app);
		myfile << mean << " " << sqrt(sumsqrd / particles.size() - mean * mean) << "\n";
		myfile.close();
	}

	

	std::clog << "\tDone with particle generation" << std::endl;
	return particles;
}