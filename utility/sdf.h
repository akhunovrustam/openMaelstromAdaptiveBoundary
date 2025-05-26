#pragma once
#include <utility/include_all.h>
#include <utility/generation/base_generation.h>

namespace sdf {
    
    struct AABB {
        float3 min;
        float3 max;

        AABB() {
            min = { std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() };
            max = { -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() };
        }

        void expand(const float3& point) {
            min.x = std::min(min.x, point.x);
            min.y = std::min(min.y, point.y);
            min.z = std::min(min.z, point.z);
            max.x = std::max(max.x, point.x);
            max.y = std::max(max.y, point.y);
            max.z = std::max(max.z, point.z);
        }

        float3 center() const {
            return { (min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f, (min.z + max.z) * 0.5f };
        }

        float longestAxis() const {
            float3 extents = { max.x - min.x, max.y - min.y, max.z - min.z };
            if (extents.x >= extents.y && extents.x >= extents.z) return 0; // x-axis
            if (extents.y >= extents.x && extents.y >= extents.z) return 1; // y-axis
            return 2; // z-axis
        }
    };

    // BVH Node
    struct BVHNode {
        AABB bounds;
        std::vector<int> triangleIndices; // Leaf node stores triangle indices
        BVHNode* left = nullptr;
        BVHNode* right = nullptr;

        ~BVHNode() {
            delete left;
            delete right;
        }

        bool isLeaf() const { return left == nullptr && right == nullptr; }
    };

    // BVH class
    class BVH {
    public:
        BVHNode* root;
        const std::vector<float3>& vertices;
        const std::vector<generation::Triangle>& triangles;

        BVH(const std::vector<float3>& verts, const std::vector<generation::Triangle>& tris)
            : vertices(verts), triangles(tris) {
            std::vector<int> indices(triangles.size());
            for (int i = 0; i < triangles.size(); ++i) indices[i] = i;
            root = buildBVH(indices, 0, indices.size());
        }

        ~BVH() { delete root; }

        // Query nearby triangles within a distance threshold
        void query(const float3& point, float radius, std::vector<int>& result) const {
            queryNode(root, point, radius, result);
        }

    private:
        BVHNode* buildBVH(std::vector<int>& indices, int start, int end) {
            BVHNode* node = new BVHNode();

            // Compute AABB for this group of triangles
            for (int i = start; i < end; ++i) {
                const auto& tri = triangles[indices[i]];
                node->bounds.expand(vertices[tri.i0]);
                node->bounds.expand(vertices[tri.i1]);
                node->bounds.expand(vertices[tri.i2]);
            }

            int count = end - start;
            if (count <= 2) { // Leaf node
                node->triangleIndices.assign(indices.begin() + start, indices.begin() + end);
                return node;
            }

            // Split along longest axis
            float axis = node->bounds.longestAxis();
            auto compare = [&](int a, int b) {
                float3 ca = triangleCentroid(triangles[a]);
                float3 cb = triangleCentroid(triangles[b]);
                return (axis == 0 ? ca.x : axis == 1 ? ca.y : ca.z) <
                       (axis == 0 ? cb.x : axis == 1 ? cb.y : cb.z);
            };
            std::sort(indices.begin() + start, indices.begin() + end, compare);

            int mid = start + count / 2;
            node->left = buildBVH(indices, start, mid);
            node->right = buildBVH(indices, mid, end);

            return node;
        }

        float3 triangleCentroid(const generation::Triangle& tri) const {
            float3 v0 = vertices[tri.i0], v1 = vertices[tri.i1], v2 = vertices[tri.i2];
            return { (v0.x + v1.x + v2.x) / 3.0f, (v0.y + v1.y + v2.y) / 3.0f, (v0.z + v1.z + v2.z) / 3.0f };
        }

        float distanceToAABB(const float3& point, const AABB& aabb) const {
            float3 closest = point;
            closest.x = std::max(aabb.min.x, std::min(closest.x, aabb.max.x));
            closest.y = std::max(aabb.min.y, std::min(closest.y, aabb.max.y));
            closest.z = std::max(aabb.min.z, std::min(closest.z, aabb.max.z));
            float dx = point.x - closest.x, dy = point.y - closest.y, dz = point.z - closest.z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        void queryNode(BVHNode* node, const float3& point, float radius, std::vector<int>& result) const {
            if (!node) return;

            float distToBox = distanceToAABB(point, node->bounds);
            if (distToBox > radius) return; // Skip if too far

            if (node->isLeaf()) {
                result.insert(result.end(), node->triangleIndices.begin(), node->triangleIndices.end());
            } else {
                queryNode(node->left, point, radius, result);
                queryNode(node->right, point, radius, result);
            }
        }
    };
    // Calculate distance from a point to a triangle
    float distanceToTriangle(float3 point, std::vector<float3> verticies, const generation::Triangle& triangle) {
        //// Calculate vectors from the vertices of the triangle to the point
        //auto v0ToPoint = point - verticies[triangle.i0];
        //auto v1ToPoint = point - verticies[triangle.i1];
        //auto v2ToPoint = point - verticies[triangle.i2];

        //// Calculate the normal of the triangle
        //auto normal = math::cross(verticies[triangle.i1] - verticies[triangle.i0], verticies[triangle.i2] - verticies[triangle.i0]);
        //normal = normal / math::length3(normal);

        //// Project the vectors onto the triangle's plane
        //float d0 = math::dot(normal, v0ToPoint);
        //float d1 = math::dot(normal, v1ToPoint);
        //float d2 = math::dot(normal, v2ToPoint);

        //// Check if the point is outside any of the triangle edges
        //float edge0 = math::dot(normal, math::cross(verticies[triangle.i1] - verticies[triangle.i0], v0ToPoint));
        //float edge1 = math::dot(normal, math::cross(verticies[triangle.i2] - verticies[triangle.i1], v1ToPoint));
        //float edge2 = math::dot(normal, math::cross(verticies[triangle.i0] - verticies[triangle.i2], v2ToPoint));

        //if (edge0 >= 0 && edge1 >= 0 && edge2 >= 0) {
        //    // Point is inside the triangle, return negative distance
        //    return -std::min(std::min(d0, d1), d2);
        //}

        //// Point is outside the triangle, return the minimum distance to its edges
        //return std::min(std::min(math::length3(v0ToPoint), math::length3(v1ToPoint)), math::length3(v2ToPoint));

        // Compute vectors from the triangle vertices to the point
        auto edge0 = verticies[triangle.i1] - verticies[triangle.i0];
        auto edge1 = verticies[triangle.i2] - verticies[triangle.i0];
        auto edge2 = verticies[triangle.i2] - verticies[triangle.i1];
        auto v0p = point - verticies[triangle.i0];

        auto normal = math::cross(edge0, edge1);
        normal = normal / math::length3(normal);

        auto cosa = math::dot(v0p, normal) / math::length3(v0p);

        auto projlengh = math::length3(v0p) * cosa;
        auto projvec = -projlengh * normal;

        auto projpoint = point + projvec; 
        //std::cout << "projpoint " << projpoint << " triangle " << verticies[triangle.i0] << " " << verticies[triangle.i1] << " " << verticies[triangle.i2] << "\n";
        //determine the vector of orientation
        auto v1 = -edge0 / math::length3(edge0) - edge1 / math::length3(edge1);
        auto v2 = edge0 / math::length3(edge0) - edge2 / math::length3(edge2);
        auto v3 = edge1 / math::length3(edge1) + edge2 / math::length3(edge2);
        
        //clockwise or opposite
        auto f1 = math::dot(math::cross(v1, verticies[triangle.i0] - projpoint), normal);
        auto f2 = math::dot(math::cross(v2, verticies[triangle.i1] - projpoint), normal);
        auto f3 = math::dot(math::cross(v3, verticies[triangle.i2] - projpoint), normal);

        bool outside = false;
        /*if ((f1 > 0 && f2 < 0) || (f1 < 0 && f2 > 0))
        {
            if (math::dot(math::cross(verticies[triangle.i0] - projpoint, verticies[triangle.i1]) - projpoint, normal) < 0)
                outside = true;
        }
        std::cout << "outside " << outside << " " << f1 << " " << f2 << " " << f3 << std::endl;
        if ((f1 > 0 && f3 < 0) || (f1 < 0 && f3 > 0))
        {
            if (math::dot(math::cross(verticies[triangle.i2] - projpoint, verticies[triangle.i0] - projpoint), normal) < 0)
                outside = true;
        }
        std::cout << "outside " << outside << std::endl;
        if ((f3 > 0 && f2 < 0) || (f3 < 0 && f2 > 0))
        {
            if (math::dot(math::cross(verticies[triangle.i1] - projpoint, verticies[triangle.i2] - projpoint), normal) < 0)
                outside = true;
        }
        std::cout << "outside " << outside << std::endl;*/

        auto w = projpoint - verticies[triangle.i0];

        float uu = math::dot(edge0, edge0);
        float uv = math::dot(edge0, edge1);
        float vv = math::dot(edge1, edge1);
        float wu = math::dot(w, edge0);
        float wv = math::dot(w, edge1);

        // Calculate the determinant
        float denominator = uv * uv - uu * vv;
        float s = (uv * wv - vv * wu) / denominator;
        float t = (uv * wu - uu * wv) / denominator;

        // Check if the point is inside the triangle
        outside = !(s >= 0 && t >= 0 && s + t <= 1);
        //std::cout << "outside " << outside << " " << s << " " << t << std::endl;

        if (outside)
        {
            float maxsmall = -1000000.f;
            float len1 = maxsmall, len2 = maxsmall, len3 = maxsmall;
            auto R1 = math::cross(math::cross(verticies[triangle.i1] - projpoint, verticies[triangle.i0] - projpoint), verticies[triangle.i1] - verticies[triangle.i0]);

            auto cosg1 = math::dot(verticies[triangle.i0] - projpoint, R1) / (math::length3(verticies[triangle.i0] - projpoint) * math::length3(R1));
            auto projr1len = math::length3(verticies[triangle.i0] - projpoint) * cosg1;
            auto projr1vec = projr1len * R1 / math::length3(R1);
            auto projr1point = projpoint + projr1vec;
            auto t1 = math::dot(projr1point - verticies[triangle.i0], verticies[triangle.i1] - verticies[triangle.i0]) / (math::length3(verticies[triangle.i1] - verticies[triangle.i0]) * math::length3(verticies[triangle.i1] - verticies[triangle.i0]));
            
            if (math::length3(R1) == 0)
            {
                projr1point = projpoint;
                t1 = -1;
                projr1len = maxsmall;
            }
            
            if (t1 >= 0 && t1 <= 1) len1 = sqrt(projr1len * projr1len + projlengh * projlengh);
            
            //std::cout << "t1 " << t1 << "\n";
            
            auto R2 = math::cross(math::cross(verticies[triangle.i2] - projpoint, verticies[triangle.i1] - projpoint), verticies[triangle.i2] - verticies[triangle.i1]);
            auto cosg2 = math::dot(verticies[triangle.i1] - projpoint, R2) / (math::length3(verticies[triangle.i1] - projpoint) * math::length3(R2));
            auto projr2len = math::length3(verticies[triangle.i1] - projpoint) * cosg2;
            auto projr2vec = projr2len * R2 / math::length3(R2);
            auto projr2point = projpoint + projr2vec;
            auto t2 = math::dot(projr2point - verticies[triangle.i1], verticies[triangle.i2] - verticies[triangle.i1]) / (math::length3(verticies[triangle.i2] - verticies[triangle.i1]) * math::length3(verticies[triangle.i2] - verticies[triangle.i1]));

            if (t2 >= 0 && t2 <= 1) len2 = sqrt(projr2len * projr2len + projlengh * projlengh);

            auto R3 = math::cross(math::cross(verticies[triangle.i0] - projpoint, verticies[triangle.i2] - projpoint), verticies[triangle.i0] - verticies[triangle.i2]);
            auto cosg3 = math::dot(verticies[triangle.i2] - projpoint, R3) / (math::length3(verticies[triangle.i2] - projpoint) * math::length3(R3));
            auto projr3len = math::length3(verticies[triangle.i2] - projpoint) * cosg3;
            auto projr3vec = projr3len * R3 / math::length3(R3);
            auto projr3point = projpoint + projr3vec;
            auto t3 = math::dot(projr3point - verticies[triangle.i2], verticies[triangle.i0] - verticies[triangle.i2]) / (math::length3(verticies[triangle.i0] - verticies[triangle.i2]) * math::length3(verticies[triangle.i0] - verticies[triangle.i2]));

            if (t3 >= 0 && t3 <= 1) len3 = sqrt(projr3len * projr3len + projlengh * projlengh);

            float returnsmall = len1;
            if (abs(returnsmall) > abs(len2)) returnsmall = len2;
            if (abs(returnsmall) > abs(len3)) returnsmall = len3;
            /*std::cout << "projRpoint " << projr1point << " " << t1 << " " << projr2point << " " << t2 << " " << projr3point << " " << t3 << " " << \
                returnsmall << " " << len1 << " " << len2 << " " << len3 << "\n";*/

            auto A1 = math::length3(point - verticies[triangle.i0]);
            auto A2 = math::length3(point - verticies[triangle.i1]);
            auto A3 = math::length3(point - verticies[triangle.i2]);

            if (abs(returnsmall) > abs(A1)) returnsmall = A1;
            if (abs(returnsmall) > abs(A2)) returnsmall = A2;
            if (abs(returnsmall) > abs(A3)) returnsmall = A3;
            
            return returnsmall;
        }
        else {
        //std::cout << "len1\n";
         
            return projlengh;
        }

        
    }

    


    

    

    // Convert mesh to SDF using grid-based representation
    float* meshToSDF(std::vector<float3> vertices, const std::vector<generation::Triangle>& triangles, 
                float gridSize, int resolution, float3 minpoint) {
        
        // Build BVH
        BVH bvh(vertices, triangles);

        // Initialize OpenVDB
        openvdb::initialize();

        // Grid setup
        int gridExtension = 10;
        int totalResolution = resolution + gridExtension * 2;
        float* sdfGrid = new float[totalResolution * totalResolution * totalResolution];
        float shift = gridSize * gridExtension;

        // Convert to OpenVDB format
        std::vector<openvdb::Vec3s> vdbVertices;
        std::vector<openvdb::Vec3I> vdbIndices;
        std::vector<openvdb::Vec4I> vdbIndices2; // Empty for triangles only
        for (const auto& v : vertices) {
            vdbVertices.push_back(openvdb::Vec3s(v.x, v.y, v.z));
        }
        for (const auto& t : triangles) {
            vdbIndices.push_back(openvdb::Vec3I(t.i0, t.i1, t.i2));
        }

        // OpenVDB transform: Use gridSize as voxel size
        openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform(gridSize);
        openvdb::Vec3d gridOrigin(minpoint.x - shift, minpoint.y - shift, minpoint.z - shift);
        xform->postTranslate(gridOrigin);

        // Compute bandwidth to cover the entire grid
        float bandwidth = gridSize * (resolution / 2.0f + gridExtension);
        openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(
            *xform, vdbVertices, vdbIndices, vdbIndices2, 
            bandwidth / gridSize, bandwidth / gridSize); // Same for interior/exterior

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*grid);

        float minDistance = std::numeric_limits<float>::infinity();

        
        // Compute SDF for each grid point
        #pragma omp parallel for collapse(3)
        for (int z = 0; z < totalResolution; ++z) {
            for (int y = 0; y < totalResolution; ++y) {
                for (int x = 0; x < totalResolution; ++x) {
                    float3 point{ minpoint.x - shift + x * gridSize, 
                                  minpoint.y - shift + y * gridSize, 
                                  minpoint.z - shift + z * gridSize };
                    float minDistance = std::numeric_limits<float>::infinity();

                    // BVH-based SDF
                   /* std::vector<int> nearbyTriangles;
                    float radius = gridSize * 20.0f;
                    bvh.query(point, radius, nearbyTriangles);
                    for (int idx : nearbyTriangles) {
                        float distance = distanceToTriangle(point, vertices, triangles[idx]);
                        if (std::abs(distance) < std::abs(minDistance)) {
                            minDistance = distance;
                        }
                    }*/

                    // OpenVDB SDF
                    openvdb::Vec3d worldPoint(point.x, point.y, point.z);
                    float distance1 = sampler.wsSample(worldPoint);

                    // Store BVH result and print comparison 
                    sdfGrid[x + y * totalResolution + z * totalResolution * totalResolution] = distance1;
                    /*if (round(minDistance) != round(distance1))
                        std::cout << "point " << point.x << " " << point.y << " " << point.z <<  " distold " << minDistance << " distvdb " << distance1 << std::endl;
                */}
            }
            std::cout << "point z=" << z << std::endl;
        }

        return sdfGrid;
    }


    // Trilinear interpolation
    hostDeviceInline float trilinearInterpolation1(float v000, float v001, float v010, float v011, float v100, float v101, float v110, float v111, float tx, float ty, float tz) {
        float tx1 = 1.0f - tx;
        float ty1 = 1.0f - ty;
        float tz1 = 1.0f - tz;
        return v000 * tx1 * ty1 * tz1 + v100 * tx * ty1 * tz1 + v010 * tx1 * ty * tz1 + v110 * tx * ty * tz1 +
            v001 * tx1 * ty1 * tz + v101 * tx * ty1 * tz + v011 * tx1 * ty * tz + v111 * tx * ty * tz;
    }
    // Lookup signed distance at an arbitrary point
    hostDeviceInline float lookupSDF1(float* sdfGrid, float3 point, float gridSize, int resolution, float3 minpoint) {
        // Convert point to grid coordinates



        resolution += 20;
        
        auto pntx = point.x - minpoint.x;
        auto pnty = point.y - minpoint.y;
        auto pntz = point.z - minpoint.z;
        // Convert point to grid coordinates
        int x = static_cast<int>(floor(pntx / gridSize));
        int y = static_cast<int>(floor(pnty / gridSize));
        int z = static_cast<int>(floor(pntz / gridSize));

        //std::cout << "xyz " << x << " " << y << " " << z << " " << x * gridSize << " " << gridSize << " " << minpoint << "\n";
        // Clamp coordinates to grid bounds
        x = std::max(0, std::min(x, resolution - 1));
        y = std::max(0, std::min(y, resolution - 1));
        z = std::max(0, std::min(z, resolution - 1));

        // Calculate grid cell indices and interpolation weights
        int x0 = x;
        int y0 = y;
        int z0 = z;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;
        float tx = pntx / gridSize - x0;
        float ty = pnty / gridSize - y0;
        float tz = pntz / gridSize - z0;

        //printf("xyz %d %d %d\n", x, y, z);
        // Retrieve signed distances at surrounding grid points
        float v000 = sdfGrid[x0 + y0 * resolution + z0 * resolution * resolution];
        float v001 = sdfGrid[x0 + y0 * resolution + z1 * resolution * resolution];
        float v010 = sdfGrid[x0 + y1 * resolution + z0 * resolution * resolution];
        float v011 = sdfGrid[x0 + y1 * resolution + z1 * resolution * resolution];
        float v100 = sdfGrid[x1 + y0 * resolution + z0 * resolution * resolution];
        float v101 = sdfGrid[x1 + y0 * resolution + z1 * resolution * resolution];
        float v110 = sdfGrid[x1 + y1 * resolution + z0 * resolution * resolution];
        float v111 = sdfGrid[x1 + y1 * resolution + z1 * resolution * resolution];

        float v00 = v000 * (1 - tx) + v100 * tx;
        float v01 = v001 * (1 - tx) + v101 * tx;
        float v10 = v010 * (1 - tx) + v110 * tx;
        float v11 = v011 * (1 - tx) + v111 * tx;

        float v0 = v00 * (1 - ty) + v10 * ty;
        float v1 = v01 * (1 - ty) + v11 * ty;

        float v = v0 * (1 - tz) + v1 * tz;

        /*std::cout << "vets " << v000 << " " << v001 << " " << v010 << " " << v011 << " " << v100 << " " << v101 << " " << v110 << " " << v111 << "\n";
        std::cout << "tx " << tx << " " << ty << " " << tz << "\n";
        std::cout << "x0 " << x0 << " " << y0 << " " << z0 << "\n";
        */// Perform trilinear interpolation
        return v;
    }
    // Perform gradient descent to find the projection of an arbitrary point onto the mesh
    hostDeviceInline float3 projectOntoMesh1(float* sdfGrid, float3 arbitraryPoint, float gridSize, int resolution, float epsilon, int maxIterations = 10, float3 minpoint = {0.f, 0.f, 0.f}) {
        float3 currentPoint = arbitraryPoint;
        float gradientStep = 0.02;
        float prevdist = 1000000.f;
        int cntr = 5;
        for (int i = 0; i < maxIterations; ++i) {
            // Lookup signed distance at current point
            float signedDistance = lookupSDF1(sdfGrid, currentPoint, gridSize, resolution, minpoint);
            prevdist = signedDistance;
            float sng = abs(signedDistance) / signedDistance;
            printf("dist dist %f\n", signedDistance);
            // Terminate if within epsilon distance of surface
            if (std::abs(signedDistance) < epsilon) {
                break;
            }

            // Compute gradient direction (negative of the gradient of SDF)
            float3 gradient{
                (math::abs(lookupSDF1(sdfGrid, currentPoint + float3{ epsilon, 0, 0 }, gridSize, resolution, minpoint)) - math::abs(signedDistance)) / epsilon,
                (math::abs(lookupSDF1(sdfGrid, currentPoint + float3{ 0, epsilon, 0 }, gridSize, resolution, minpoint)) - math::abs(signedDistance)) / epsilon,
                (math::abs(lookupSDF1(sdfGrid, currentPoint + float3{0, 0, epsilon}, gridSize, resolution, minpoint)) - math::abs(signedDistance)) / epsilon
            };

            std::cout << "gradient " << gradient << "\n";
            std::cout << "gradientx " << math::abs(lookupSDF1(sdfGrid, currentPoint + float3{ epsilon, 0, 0 }, gridSize, resolution, minpoint)) 
                << " " << math::abs(signedDistance) << " " << (math::abs(lookupSDF1(sdfGrid, currentPoint + float3{ epsilon, 0, 0 }, gridSize, resolution, minpoint)) - math::abs(signedDistance)) / epsilon << "\n";
            
            // Update current point using gradient descent
            currentPoint = currentPoint - gradient * (gradientStep / math::length3(gradient));
            signedDistance = lookupSDF1(sdfGrid, currentPoint, gridSize, resolution, minpoint);

            std::cout << "curpoint " << currentPoint << "\n";
            if (abs(prevdist) < abs(signedDistance)) {
                currentPoint = currentPoint + gradient * (gradientStep / math::length3(gradient));
                signedDistance = prevdist;
                cntr--;
                gradientStep /= 1.5;
            }
            std::cout << "step " << gradientStep << "\n";
            if (cntr == 0) break;

            prevdist = signedDistance;
        }
        return currentPoint;
    }
}