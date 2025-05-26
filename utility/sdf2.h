#pragma once
#include <utility/include_all.h>
//#include <utility/generation/base_generation.h>

namespace sdf {
    
    // Trilinear interpolation
    hostDeviceInline float trilinearInterpolation(float v000, float v001, float v010, float v011, float v100, float v101, float v110, float v111, float tx, float ty, float tz) {
        float tx1 = 1.0f - tx;
        float ty1 = 1.0f - ty;
        float tz1 = 1.0f - tz;
        return v000 * tx1 * ty1 * tz1 + v100 * tx * ty1 * tz1 + v010 * tx1 * ty * tz1 + v110 * tx * ty * tz1 +
            v001 * tx1 * ty1 * tz + v101 * tx * ty1 * tz + v011 * tx1 * ty * tz + v111 * tx * ty * tz;
    }
    // Lookup signed distance at an arbitrary point
    hostDeviceInline float lookupSDF(float* sdfGrid, float3 point, float gridSize, int resolution, float4 minpoint) {
        // Convert point to grid coordinates
        resolution += 20;

        auto pntx = point.x - minpoint.x;
        auto pnty = point.y - minpoint.y;
        auto pntz = point.z - minpoint.z;
        // Convert point to grid coordinates
        int x = static_cast<int>(floor(pntx / gridSize));
        int y = static_cast<int>(floor(pnty / gridSize));
        int z = static_cast<int>(floor(pntz / gridSize));

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

        // Perform trilinear interpolation
        return v;
    }
    // Perform gradient descent to find the projection of an arbitrary point onto the mesh
    hostDeviceInline float3 projectOntoMesh(float* sdfGrid, float3 arbitraryPoint, float radius, float gridSize, int resolution, float epsilon, int maxIterations = 10, float4 minpoint = {0.f, 0.f, 0.f, 0.f}, int ind = 0) {
        float3 currentPoint = arbitraryPoint;
        float gradientStep = 0.05 * radius;
        float prevdist = 1000000.f;
        int cntr = 5;
        for (int i = 0; i < maxIterations; ++i) {
            // Lookup signed distance at current point
            float signedDistance = lookupSDF(sdfGrid, currentPoint, gridSize, resolution, minpoint);
            prevdist = signedDistance;
            float sng = abs(signedDistance) / signedDistance;
            //printf("dist dist %f\n", signedDistance);
            // Terminate if within epsilon distance of surface
            if (std::abs(signedDistance) < epsilon) {
                //printf("BREAk ind %d dist %f eps %f\n", ind, std::abs(signedDistance), epsilon);
                break;
            }

            // Compute gradient direction (negative of the gradient of SDF)
            float3 gradient{
                (math::abs(lookupSDF(sdfGrid, currentPoint + float3{ epsilon, 0, 0 }, gridSize, resolution, minpoint)) - math::abs(signedDistance)) / epsilon,
                (math::abs(lookupSDF(sdfGrid, currentPoint + float3{ 0, epsilon, 0 }, gridSize, resolution, minpoint)) - math::abs(signedDistance)) / epsilon,
                (math::abs(lookupSDF(sdfGrid, currentPoint + float3{0, 0, epsilon}, gridSize, resolution, minpoint)) - math::abs(signedDistance)) / epsilon
            };

            //std::cout << "gradient " << gradient << "\n";
            //std::cout << "gradientx " << math::abs(lookupSDF1(sdfGrid, currentPoint + float3{ epsilon, 0, 0 }, gridSize, resolution, minpoint)) 
            //    << " " << math::abs(signedDistance) << " " << (math::abs(lookupSDF1(sdfGrid, currentPoint + float3{ epsilon, 0, 0 }, gridSize, resolution, minpoint)) - math::abs(signedDistance)) / epsilon << "\n";
            
            // Update current point using gradient descent
            currentPoint = currentPoint - gradient * (gradientStep / math::length3(gradient));
            signedDistance = lookupSDF(sdfGrid, currentPoint, gridSize, resolution, minpoint);

            //std::cout << "curpoint " << currentPoint << "\n";
            if (abs(prevdist) < abs(signedDistance)) {
                currentPoint = currentPoint + gradient * (gradientStep / math::length3(gradient));
                signedDistance = prevdist;
                cntr--;
                gradientStep /= 1.5;
            }
            //std::cout << "step " << gradientStep << "\n";
            if (cntr == 0) break;

            prevdist = signedDistance;
        }
        return currentPoint;
    }

   

}