#include <utility/include_all.h>
#include <utility/generation.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <cfloat>
#include <utility/generation/regular_grid_centered.h>
#include <utility/generation/optimized_generation.h>
#include <numbers>

namespace generation {
grid_tuple loadVDBFile(fs::path fileName) {
  // vdb cannot work with fs paths for now
  std::string vdbFile = fileName.string();
  vdb::initialize();
  vdb::io::File file(vdbFile);
  file.open();
  auto grids = file.getGrids();
  vdb::GridBase::Ptr baseGrid;
  // Iterate over all grids to find "surface grid" which should be a SDF of float type
  for (vdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
    if (nameIter.gridName() == "surface")
      baseGrid = file.readGrid(nameIter.gridName());
  file.close();
  // Cast grid over, not our problem if the grid wasn't actually a float grid
  vdb::FloatGrid::Ptr grid = vdb::gridPtrCast<vdb::FloatGrid>(baseGrid);
  // Used to get the AABB for some other methods
  vdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
  return std::make_tuple(grid, grid->indexToWorld(box.getStart()), grid->indexToWorld(box.getEnd()));
}
grid_tuple VDBfromOBJ(fs::path fileName) {
  std::ifstream objFile(fileName);

  std::vector<vdb::Vec3s> vertices;
  std::vector<vdb::Vec3I> indices;

  for (std::string line; std::getline(objFile, line);) {
    auto vs = readValues<float>(line, "v");
    if (vs.size() != 0) {
      if (vs.size() != 3)
        throw std::invalid_argument(std::string("could not parse line ") + line +
                                    " as it did not contain 3 coordinates.");
      vertices.push_back(vdb::Vec3s(vs[0], vs[1], vs[2]));
      continue;
    }
    auto fs = readValues<int32_t>(line, "f");
    if (fs.size() != 0) {
      if (fs.size() != 3 && fs.size() != 4)
        throw std::invalid_argument(std::string("could not parse line ") + line +
                                    " as it did not contain 3 or 4 indices.");
      indices.push_back(vdb::Vec3I(fs[0] - 1, fs[1] - 1, fs[2] - 1));
      if (fs.size() == 4)
        indices.push_back(vdb::Vec3I(fs[2] - 1, fs[3] - 1, fs[0] - 1));
      continue;
    }
    // Skipping line
  }

  const float voxelSize = 0.5f;
  vdbm::Transform::Ptr xform = vdbm::Transform::createLinearTransform(voxelSize);
  auto grid = vdbt::meshToLevelSet<vdb::FloatGrid>(*xform, vertices, indices);
  vdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
  return std::make_tuple(grid, grid->indexToWorld(box.getStart()), grid->indexToWorld(box.getEnd()));
}
grid_tuple VDBfromPly(fs::path fileName) {
  std::ifstream objFile(fileName);

  std::vector<vdb::Vec3s> vertices;
  std::vector<vdb::Vec3I> indices;

  int32_t vertexIdx = 0;
  int32_t triangleIdx = 0;

  std::string line;
  unsigned totalVertices, totalTriangles, lineNo = 0;
  bool inside = false;
  while (std::getline(objFile, line)) {
    lineNo++;
    if (!inside) {
      if (line.substr(0, 14) == "element vertex") {
        std::istringstream str(line);
        std::string word1;
        str >> word1;
        str >> word1;
        str >> totalVertices;
        vertices.resize(totalVertices);
      } else if (line.substr(0, 12) == "element face") {
        std::istringstream str(line);
        std::string word1;
        str >> word1;
        str >> word1;
        str >> totalTriangles;
        indices.resize(totalTriangles);
      } else if (line.substr(0, 10) == "end_header")
        inside = true;
    } else {
      if (totalVertices) {

        totalVertices--;
        float x, y, z;

        std::istringstream str_in(line);
        str_in >> x >> y >> z;
        auto &pCurrentVertex = vertices[vertexIdx];
        pCurrentVertex.x() = x;
        pCurrentVertex.y() = y;
        pCurrentVertex.z() = z;
        vertexIdx++;
      }

      else if (totalTriangles) {

        totalTriangles--;
        unsigned dummy;
        unsigned idx1, idx2, idx3; // vertex index
        std::istringstream str2(line);
        if (str2 >> dummy >> idx1 >> idx2 >> idx3) {
          auto &pCurrentTriangle = indices[vertexIdx];
          pCurrentTriangle.x() = idx1;
          pCurrentTriangle.y() = idx2;
          pCurrentTriangle.z() = idx3;
          triangleIdx++;
        }
      }
    }
  }
  const float voxelSize = 0.5f;
  vdbm::Transform::Ptr xform = vdbm::Transform::createLinearTransform(voxelSize);
  auto grid = vdbt::meshToLevelSet<vdb::FloatGrid>(*xform, vertices, indices);
  vdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
  return std::make_tuple(grid, grid->indexToWorld(box.getStart()), grid->indexToWorld(box.getEnd()));
}
grid_tuple fileToVDB(fs::path path) {
  if (path.extension() == ".vdb")
    return loadVDBFile(path);
  if (path.extension() == ".obj")
    return VDBfromOBJ(path);
  if (path.extension() == ".ply")
    return VDBfromOBJ(path);
  else {
    std::cerr << "Unknown file extension in " << path.string() << std::endl;
    throw std::invalid_argument("Unknown extension");
  }
}

obj_tuple ObjFromObj(fs::path path) {
  std::ifstream objFile(path);

  std::vector<float3> vertices;
  std::vector<Triangle> planes;
  std::vector<Edge> edges;

  for (std::string line; std::getline(objFile, line);) {
    auto vs = readValues<float>(line, "v");
    if (vs.size() != 0) {
      if (vs.size() != 3)
        throw std::invalid_argument(std::string("could not parse line ") + line +
                                    " as it did not contain 3 coordinates.");
      vertices.push_back(float3{vs[0], vs[1], vs[2]});
      continue;
    }
    auto fs = readValues<int32_t>(line, "f");
    if (fs.size() != 0) {
      if (fs.size() != 3 && fs.size() != 4)
        throw std::invalid_argument(std::string("could not parse line ") + line +
                                    " as it did not contain 3 or 4 indices.");
      planes.push_back(Triangle{fs[0] - 1, fs[1] - 1, fs[2] - 1});
      if (fs.size() == 4)
        planes.push_back(Triangle{fs[2] - 1, fs[3] - 1, fs[0] - 1});
      continue;
    }
  }
  for (const auto &plane : planes) {
    edges.push_back(Edge{plane.i0, plane.i1});
    edges.push_back(Edge{plane.i1, plane.i2});
    edges.push_back(Edge{plane.i2, plane.i0});
  }

  float3 min = vector_t<float, 3>::max();
  float3 max = vector_t<float, 3>::min();
  for (const auto &vtx : vertices) {
    min = math::min(min, vtx);
    max = math::max(max, vtx);
  }

  return std::make_tuple(vertices, planes, edges, min, max);
}
obj_tuple ObjFromPly(fs::path path) {
  std::ifstream objFile(path);

  std::vector<float3> vertices;
  std::vector<Triangle> planes;
  std::vector<Edge> edges;

  int32_t vertexIdx = 0;
  int32_t triangleIdx = 0;

  std::string line;
  unsigned totalVertices, totalTriangles, lineNo = 0;
  bool inside = false;
  while (std::getline(objFile, line)) {
    lineNo++;
    if (!inside) {
      if (line.substr(0, 14) == "element vertex") {
        std::istringstream str(line);
        std::string word1;
        str >> word1;
        str >> word1;
        str >> totalVertices;
        vertices.resize(totalVertices);
      } else if (line.substr(0, 12) == "element face") {
        std::istringstream str(line);
        std::string word1;
        str >> word1;
        str >> word1;
        str >> totalTriangles;
        planes.resize(totalTriangles);
      } else if (line.substr(0, 10) == "end_header")
        inside = true;
    } else {
      if (totalVertices) {

        totalVertices--;
        float x, y, z;

        std::istringstream str_in(line);
        str_in >> x >> y >> z;
        auto &pCurrentVertex = vertices[vertexIdx];
        pCurrentVertex.x = x;
        pCurrentVertex.y = y;
        pCurrentVertex.z = z;
        vertexIdx++;
      }

      else if (totalTriangles) {

        totalTriangles--;
        unsigned dummy;
        unsigned idx1, idx2, idx3; // vertex index
        std::istringstream str2(line);
        if (str2 >> dummy >> idx1 >> idx2 >> idx3) {
          auto &pCurrentTriangle = planes[triangleIdx];
          pCurrentTriangle.i0 = idx1;
          pCurrentTriangle.i1 = idx2;
          pCurrentTriangle.i2 = idx3;
          triangleIdx++;
        }
      }
    }
  }
  for (const auto &plane : planes) {
    edges.push_back(Edge{plane.i0, plane.i1});
    edges.push_back(Edge{plane.i1, plane.i2});
    edges.push_back(Edge{plane.i2, plane.i0});
  }

  float3 min = vector_t<float, 3>::max();
  float3 max = vector_t<float, 3>::min();
  for (const auto &vtx : vertices) {
    min = math::min(min, vtx);
    max = math::max(max, vtx);
  }

  return std::make_tuple(vertices, planes, edges, min, max);

}
obj_tuple ObjFromVDB(fs::path path) {
  std::vector<uint32_t> indices;
  std::vector<vdb::Vec3f> pyramidVertices;
  std::vector<vdb::Vec3f> pyramidNormals;
  auto [grid, min_vdb, max_vdb] = fileToVDB(path);
  float3 min{(float)min_vdb.x(), (float)min_vdb.y(), (float)min_vdb.z()};
  float3 max{(float)max_vdb.x(), (float)max_vdb.y(), (float)max_vdb.z()};

  std::vector<float3> vertices;
  std::vector<Triangle> planes;
  std::vector<Edge> edges;

  vdbt::VolumeToMesh mesher(grid->getGridClass() == vdb::GRID_LEVEL_SET ? 0.0 : 0.01, 1.0);
  mesher(*grid);

  for (vdb::Index64 n = 0, i = 0, N = mesher.pointListSize(); n < N; ++n) {
    const vdb::Vec3s &p = mesher.pointList()[n];
    vertices.push_back(float3{p.x(), p.y(), p.z()});
  }
  vdbt::PolygonPoolList &polygonPoolList = mesher.polygonPoolList();
  for (vdb::Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
    const vdbt::PolygonPool &polygons = polygonPoolList[n];
    for (vdb::Index64 i = 0, I = polygons.numQuads(); i < I; ++i) {
      const vdb::Vec4I &quad = polygons.quad(i);
      auto i0 = (int32_t)quad[0];
      auto i1 = (int32_t)quad[1];
      auto i2 = (int32_t)quad[2];
      auto i3 = (int32_t)quad[3];
      planes.push_back(Triangle{i0, i1, i2});
      planes.push_back(Triangle{i2, i3, i0});
      edges.push_back(Edge{i0, i1});
      edges.push_back(Edge{i1, i2});
      edges.push_back(Edge{i2, i3});
      edges.push_back(Edge{i3, i0});
      edges.push_back(Edge{i0, i2});
    }
    for (vdb::Index64 i = 0, I = polygons.numTriangles(); i < I; ++i) {
      const vdb::Vec3I &quad = polygons.triangle(i);
      auto i0 = (int32_t)quad[0];
      auto i1 = (int32_t)quad[1];
      auto i2 = (int32_t)quad[2];
      planes.push_back(Triangle{i0, i1, i2});
      edges.push_back(Edge{i0, i1});
      edges.push_back(Edge{i1, i2});
      edges.push_back(Edge{i2, i0});
    }
  }
  return std::make_tuple(vertices, planes, edges, min, max);
}
obj_tuple fileToObj(fs::path path) {
  if (path.extension() == ".vdb")
    return ObjFromVDB(path);
  if (path.extension() == ".obj")
    return ObjFromObj(path);
  if (path.extension() == ".ply")
    return ObjFromPly(path);
  else {
    std::cerr << "Unknown file extension in " << path.string() << std::endl;
    throw std::invalid_argument("Unknown extension");
  }
}

std::vector<vdb::Vec4f> generateParticles(std::string fileName, float r, genTechnique kind, bool clampToDomain, float3 scale) {
  auto path = resolveFile(fileName, {get<parameters::internal::config_folder>()});
  std::vector<vdb::Vec4f> particles;
  if (kind == genTechnique::hex_grid || kind == genTechnique::square_grid) {
    auto [grid, min_vdb, max_vdb] = fileToVDB(path);
    if (kind == genTechnique::hex_grid)
      return VolumeToHex(grid, min_vdb, max_vdb, r, scale);
    else
      return VolumeToRegular(grid, min_vdb, max_vdb, r);
  } else {
    return ObjToShell(path, r);
  }
}

std::vector<vdb::Vec4f> generateParticlesRigid(std::string fileName, float r, genTechnique kind, bool clampToDomain, float3 scale, float sampling) {
  auto path = resolveFile(fileName, {get<parameters::internal::config_folder>()});
  std::vector<vdb::Vec4f> particles;
  if (kind == genTechnique::hex_grid || kind == genTechnique::square_grid || kind == genTechnique::optimized_density || kind == genTechnique::optimized_impulse 
      || kind == genTechnique::random) {
    auto [grid, min_vdb, max_vdb] = fileToVDB(path);
    if (kind == genTechnique::hex_grid)
      return VolumeToHex(grid, min_vdb, max_vdb, r);
    else if ( kind == genTechnique::optimized_impulse) {
        return sampleOptimized(path.string());
    }
    else if ( kind == genTechnique::optimized_density) {
        return sampleOptimizedDens(path.string());
    }
    else if ( kind == genTechnique::random) {
        return RandomSampling(grid, min_vdb, max_vdb, r, scale, sampling);
    }
    else
      return VolumeToRegularCentered(grid, min_vdb, max_vdb, r, scale, sampling);
  } else {
    return ObjToShell(path, r);
  }
}

std::vector<vdb::Vec4f> ImpulseOptimize(float vol, std::vector<vdb::Vec4f> parts) {
    return impulseOptimize(vol, parts);
}

std::vector<vdb::Vec4f> DensityOptimize(float vol, std::vector<vdb::Vec4f> parts) {
    return densityOptimize(vol, parts);
}

std::vector<vdb::Vec4f> RandomSampling(vdb::FloatGrid::Ptr grid, vdb::Vec3d min_vdb, vdb::Vec3d max_vdb, 
  float r, float3 scale, float sampling)
{
    std::vector<vdb::Vec4f> particles;
    float density = 2.f;
    std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-10.0, 10.0);
	std::uniform_real_distribution<> dis1(0.0, 1.0);
    
    min_vdb = vdb::Vec3d({ -15.f, -15.f, 0.f });
    max_vdb = vdb::Vec3d({ 15.f, 15.f, 0.f });

	auto a = std::numbers::pi * r * r;
	auto area = density * (min_vdb.x() - max_vdb.x()) * (min_vdb.y() - max_vdb.y());
    vdb::Vec3d p1 = vdb::Vec3d({ 15.f, -15.f, max_vdb.z() });
    vdb::Vec3d p2 = vdb::Vec3d({ -15.f, 15.f, max_vdb.z() });
	auto ptcls = static_cast<int32_t>(area / a) + (dis1(gen) < area / a - std::floor(area / a) ? 1 : 0);
	for (int32_t i = 0; i < ptcls; ++i) {
		auto s = dis1(gen);
		auto t = dis1(gen);
		
		auto p = min_vdb + t * (p1 - min_vdb) + s * (p2 - min_vdb) /*+ (1-t) * (p2)*/;
		particles.push_back(openvdb::Vec4f{(float)p.x(), (float)p.y(), (float)p.z(), r * 2.f});
	}
    /*std::ofstream myfile;
	myfile.open ("points_dump.txt");
	for (auto pt : particles)
	{
		myfile << pt.x() << " " << pt.y() << " " << pt.z() << " " << pt.w() << "\n";
	}
	myfile.close();*/

    ///////////////////////////////////////
    std::vector<vdb::Vec4f> particles2;
    std::string line;
    std::ifstream myfile1 ("points_dump.txt");
    if (myfile1.is_open())
    {
        while ( std::getline (myfile1,line) )
        {
            std::string delimiter = " ";

            size_t pos = 0;
            std::string token;
            int posnum = 0;
            vdb::Vec4f pnt;
            while ((pos = line.find(delimiter)) != std::string::npos) {
                token = line.substr(0, pos);
                if (posnum == 0) pnt.x() = std::stof(token);
                else if (posnum == 1) pnt.y() = std::stof(token);
                else if (posnum == 2) pnt.z() = std::stof(token);
                
                posnum++;
                line.erase(0, pos + delimiter.length());
            }
            pnt.w() = std::stof(line);
            particles2.push_back(pnt);
        }
        particles = particles2;
        myfile1.close();
    }
   
    /////////////////////////////////////

    return particles;
}

} // namespace generation