#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <render/util/color_map.h>
#include <tools/log.h>
#include <utility/identifier/uniform.h>

GLuint colorMap::create1DTexture(float4 *colorMap, int32_t elements) {
  GLuint textureId_;

  // generate the specified number of texture objects
  glGenTextures(1, &textureId_);
  // assert(glGetError() == GL_NO_ERROR);

  // bind texture
  glBindTexture(GL_TEXTURE_1D, textureId_);
  // assert(glGetError() == GL_NO_ERROR);

  // tells OpenGL how the data that is going to be uploaded is aligned
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  // assert(glGetError() == GL_NO_ERROR);

  glTexImage1D(
      GL_TEXTURE_1D, // Specifies the target texture. Must be GL_TEXTURE_1D or GL_PROXY_TEXTURE_1D.
      0, // Specifies the level-of-detail number. Level 0 is the base image level. Level n is the
         // nth mipmap reduction image.
      GL_RGBA32F, elements,
      0, // border: This value must be 0.
      GL_RGBA, GL_FLOAT, colorMap);
  // assert(glGetError() == GL_NO_ERROR);

  // texture sampling/filtering operation.
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  // assert(glGetError() == GL_NO_ERROR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  // assert(glGetError() == GL_NO_ERROR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // assert(glGetError() == GL_NO_ERROR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  // assert(glGetError() == GL_NO_ERROR);

  glBindTexture(GL_TEXTURE_1D, 0);
  // assert(glGetError() == GL_NO_ERROR);

  return textureId_;
}

#include <tools/pathfinder.h>
#define STB_IMAGE_IMPLEMENTATION
#include <render/util/stb_image.h>

void colorMap::update() {
  static std::string old_colormap = "";
  static size_t old_size = 0;
  int32_t image_width = 1024;
  int32_t image_height = 1024;
  float4* img = new float4[1024];
  if (old_colormap != get<parameters::color_map::map>()) {

    old_colormap = get<parameters::color_map::map>();

    //std::string file_name = root_folder + get<parameters::color_map::map>() + ".png";
    //std::cout << file_name << " exists: " << fs::exists(file_name) << std::endl;
    for (int32_t it = 0; it < 1024; ++it)
        img[it] = float4{ (float)it / (float)1024 * 255.f,(float)it / (float)1024 * 255.f,(float)it / (float)1024 * 255.f, 255.f };
	std::string file_name = resolveFile(std::string("cfg/") + get<parameters::color_map::map>() + ".png").string();
	if (std::filesystem::exists(file_name)) {
		//std::cout << "Loading " << file_name << std::endl;
        unsigned char* image_data = stbi_load(file_name.c_str(), &image_width, &image_height, NULL, 4);
        delete[] img;
        img = new float4[image_width];
        for (int32_t it = 0; it < image_width; ++it) {
            img[it] = float4{
            (float)image_data[it * 4 + 0],
            (float)image_data[it * 4 + 1],
            (float)image_data[it * 4 + 2], 255.f };
            //std::cout << it << ": [ " << img[it].x << " " << img[it].y << " " << img[it].z <<
            //    " ]" << std::endl;
        }
		//img = QImage(QString::fromStdString(file_name));
		//img.load(QString(file_name.c_str()));
		//std::cout << image_width << " : " << image_height << std::endl;
	}
	//catch (...) {}
    color_map = (float4 *)realloc(color_map, sizeof(float4) * (image_width));
    for (int32_t it = 0; it < image_width; ++it) {
      color_map[it] = float4{(float)(img[it].x) / 256.f, (float)(img[it].y) / 256.f,
                                (float)(img[it].z) / 256.f, 1.f};
	  //std::cout << color_map[it].x() << " : " << color_map[it].y() << " : " << color_map[it].z() << std::endl;
		//if(it == img.width()-1)
		//	color_map[it + 1] = QVector4D{ (float)(col.red()) / 256.f, (float)(col.green()) / 256.f,
		//	(float)(col.blue()) / 256.f, 1.f };
    }
	
    color_map_elements = image_width;
    delete[] img;
    texunit = create1DTexture(color_map, color_map_elements);
    for (auto b : bindings) {
      auto [prog, texture_id, identifier] = b;

      //prog->bind();
      //GLuint samplerLocation = prog->uniformLocation(identifier.c_str());
      glUseProgram(prog);
      GLuint samplerLocation = glGetUniformLocation(prog, identifier.c_str());
      glUniform1i(samplerLocation, texture_id);
      glActiveTexture(GL_TEXTURE0 + texture_id);
      glBindTexture(GL_TEXTURE_1D, texunit);
      glUseProgram(0);
     // prog->release();
    }
    old_size = bindings.size();
  } else if (bindings.size() != old_size) {
    for (auto b : bindings) {
      auto [prog, texture_id, identifier] = b;
      glUseProgram(prog);
      GLuint samplerLocation = glGetUniformLocation(prog, identifier.c_str());
      glUniform1i(samplerLocation, texture_id);
      glActiveTexture(GL_TEXTURE0 + texture_id);
      glBindTexture(GL_TEXTURE_1D, texunit);
      glUseProgram(0);
    }
    old_size = bindings.size();
  }
}

void colorMap::bind(GLint prog, GLint texture_id, std::string id) {

  bindings.emplace_back(prog, texture_id, id);
}

colorMap &colorMap::instance() {
  static colorMap inst;
  return inst;
}
