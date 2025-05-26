#pragma once
#include <GLFW/glfw3.h>
#include <math/vector_types.h>
#include <math/matrix.h>

struct DeviceCamera {
	__host__ __device__ constexpr DeviceCamera() {}
	float2 resolution{ 1920, 1080 };     ///< Resolution of the viewport/image
	float3 position = { 0.f, 0.f, 0.f }; ///< Camera position ( eye position )
	float3 view = { 1.f, 0.f, 0.f };     ///< Viewing direction
	float3 up = { 0.f, 0.f, 1.f };       ///< Up direction
	float2 fov = { 24.f, 32.f };         ///< Fov of the camera in x and y
	float apertureRadius = 0.f;        ///< Aperture for DOF, unused currently
	float focalDistance = 1.f; ///< Focal distance for DOF, unused currently

	Matrix4x4 ViewInverse{}; ///< Inverted view matrix, generated on host side
	Matrix4x4 PerspInverse{}; ///< Inverted perspective matrix, generated on host side
	Matrix4x4 MVP{}; ///< VP matrix, generated on host side
};

struct SceneInformation {
	__device__ __host__ constexpr SceneInformation() {};
	DeviceCamera m_camera{};
	float width = 1920;
	float height = 1080;
	bool dirty = true;
};

#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>
#include <render/util/camera.h>
#include <variant>
#include <map>
#include <vector>
/** Polymorphic base class which should be used for all rendering functions as
 * all instances that way can be stored inside of a vector of the openGL
 * widget. **/
class Renderer {
public:
	GLuint vao;
	GLuint m_program;

  virtual void update() = 0;
  virtual void render(bool pretty = false) = 0;
  virtual bool valid() = 0;
  virtual void toggle(){};
  virtual void resize(int32_t w, int32_t h) {};

  virtual void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {}
  virtual void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {}
  virtual void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {}
  virtual void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {}
  virtual std::string getInformation() { return std::string(""); }
};
#ifndef __CUDACC__
class RTXRender : public Renderer {
protected:
  int32_t frame = -1;
  int32_t framenumber = 0;
  virtual void updateRTX() = 0;
  virtual void renderRTX(bool pretty = false, int32_t framenumber = 0, int32_t seed = 0x12345) = 0;

  GLuint defer_VAO;
  GLuint quad_programID;

  GLuint renderedTextureOut;
  cudaGraphicsResource_t renderedResourceOut;

  void prepCUDAscene();
  float3 *accumulatebuffer = nullptr;

  bool dirty = false;
  int32_t initialFrame;

public:
  bool bValid = false;
  RTXRender();
  virtual void uiFunction() {};

  virtual void update();
  virtual void render(bool pretty = false);
  virtual bool valid();
  virtual void toggle(){};
  virtual std::map<std::string, std::variant<int32_t, float, std::vector<int32_t>, std::vector<float>>>
  getData() {
    return {};
  };

};
#endif
extern SceneInformation &hostScene();

GLuint createProgram(std::string vtx, std::string frag);
