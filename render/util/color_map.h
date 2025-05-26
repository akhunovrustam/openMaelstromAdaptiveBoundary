#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <render/util/renderer.h>
/** This class is used to load color maps from files. The color maps are loaded
 * from pngs relative to the resource path. **/
class colorMap{
  GLuint create1DTexture(float4 *colors, int32_t elements);
  float4*color_map = nullptr;
  int32_t color_map_elements = 0;
  GLint texunit = -1;

  colorMap() {
    update();
  }

  std::vector<std::tuple<GLint, GLint, std::string>> bindings;

public:
  static colorMap &instance();

  void update();
  void bind(GLint prog, GLint texture_id, std::string id);
};
