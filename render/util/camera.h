#pragma once
#include <GLFW/glfw3.h>
#include <render/util/renderer.h>
#include <glm/glm.hpp>

/** This class represents the main camera of the simulation with a first person
 * style camera. The camera is built using a singleton instance for easier
 * access, i.e. in other renderers. The uniforms are created by the openGL
 * Widget.**/
struct Camera {
private:
  Camera() {};

public:

  static Camera &instance();

  enum CameraType { lookat, firstperson };

  void updateViewMatrix();
  bool moving();
  void setPerspective(float fov, float aspect, float znear, float zfar);
  void updateAspectRatio(float aspect);

  std::pair<bool, DeviceCamera> prepareDeviceCamera();

  void setPosition(glm::vec3 position);
  void setRotation(glm::vec3 rotation);
  void rotate(glm::vec3 delta);
  void setTranslation(glm::vec3 translation);
  void translate(glm::vec3 delta);
  void update(float deltaTime);
  void maximizeCoverage();

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
  void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
  void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
  void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

  float fov;
  float znear, zfar, aspect;

  int32_t width, height;

  CameraType type = CameraType::firstperson;

  glm::vec3 rotation = glm::vec3();
  glm::vec3 position = glm::vec3();
  glm::vec3 strafe = glm::vec3();
  glm::vec3 forward = glm::vec3();
  glm::vec3 up = glm::vec3();
  

  float rotationSpeed = 0.5f;
  float movementSpeed = 1.0f;

  bool tracking = false;
  bool dirty = true;
  struct {
    glm::mat4 perspective;
    glm::mat4 view;
  } matrices;

  bool lbuttondown = false;
  bool rbuttondown = false;
  bool mbuttondown = false;
  float2 mousePos;

  struct {
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;
    bool q = false;
    bool e = false;
  } keys;
};
