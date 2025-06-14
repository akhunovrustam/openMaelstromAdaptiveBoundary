#include <utility/include_all.h>
#include <utility/generation.h>
#include <iostream>

namespace generation {
std::vector<vdb::Vec4f> VolumeToRegularCentered(vdb::FloatGrid::Ptr grid, vdb::Vec3d min_vdb, vdb::Vec3d max_vdb, 
  float r, float3 scale, float sampling) {

  vdbt::GridSampler<vdb::FloatGrid, vdbt::BoxSampler> sampler(*grid);

  auto v = PI4O3 * math::power<3>(r);
  auto [spacing, h, H] = getPacking(r);
  spacing = math::power<ratio<1, 3>>(v) * sampling;

  auto [mind, maxd] = getDomain();

  auto gen_position_offset = [=, spacing = spacing](auto offset, int32_t i, int32_t j, int32_t k) {
    vdb::Vec4f initial{(float)i, (float)j, (float)k, 0.f};
    return offset + initial * spacing;
  };

  vdb::Vec4f firstPoint{(float)min_vdb.x() + 2.f * spacing, (float)min_vdb.y() + 2.f * spacing,
                        (float)min_vdb.z() + 2.f * spacing, h};

  std::vector<vdb::Vec4f> out_points;

  //float allowedlen = max_vdb.x() - min_vdb.x() - 4.f * spacing;

  //TODO: simplify generated points' shifting
  double avg_x = 0, avg_y = 0, avg_z = 0;
  int32_t i, j, k;
  i = j = k = 0; 
  vdb::Vec4f inserted;
  int32_t cnt = 0;
  do {
    do {
      do {
        inserted = gen_position_offset(firstPoint, i++, j, k);
        // if (inserted.x() < mind.x || inserted.y() < mind.y || inserted.z() < mind.z || inserted.x() > maxd.x ||
        //     inserted.y() > maxd.y || inserted.z() > maxd.z)
        //   continue;

        if (sampler.wsSample(vdb::Vec3d(inserted.x() / scale.x, inserted.y() / scale.y, inserted.z() / scale.z)) <  1)
        {
          out_points.push_back(inserted);
          avg_x += inserted.x();
          avg_y += inserted.y();
          avg_z += inserted.z();
          cnt++;
        }
      } while (inserted.x() / scale.x < max_vdb.x() - 2.f * spacing);
      i = 0;
      ++j;
    } while (inserted.y() / scale.y < max_vdb.y() - 2.f * spacing);
    j = 0;
    k++;
  } while (inserted.z() / scale.z < max_vdb.z() - 2.f * spacing);

  

  return out_points;
}

} // namespace generation