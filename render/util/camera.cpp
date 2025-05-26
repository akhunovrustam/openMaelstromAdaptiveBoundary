#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <render/util/camera.h>
#include <tools/log.h>
#include <utility/identifier/uniform.h>

Camera &Camera::instance() {
  static Camera cam;
  return cam;
}

void Camera::maximizeCoverage()
{
    auto to_vec3 = [](auto v) { return glm::vec3(v.x, v.y, v.z); };
	float3 minAABB = get<parameters::internal::min_coord>() + 3.f * get<parameters::internal::cell_size>() - get<parameters::particle_settings::radius>();
	float3 maxAABB = get<parameters::internal::max_coord>() - 3.f * get<parameters::internal::cell_size>() + get<parameters::particle_settings::radius>();
	float3 center = minAABB + (maxAABB - minAABB) * 0.5f;

	auto xv = [&](auto x, auto y, auto z, auto view) {
		glm::vec4 xp( x, y, z, 1.f );
		auto vp = view * xp;
		auto pp = matrices.perspective * vp;
		pp = pp / pp.w;
		auto pred =
			((pp.x <= 1.f) && (pp.x >= -1.f)) &&
			((pp.y <= 1.f) && (pp.y >= -1.f)) &&
			((pp.z <= 1.f) && (pp.z >= 0.f));
		return std::make_tuple(xp, pp, pred);
	};

	//glm::vec3 position = position;
	glm::vec3 fwd = forward;

	float pitch = rotation.x * M_PI / 180.f;
	float yaw = rotation.z * M_PI / 180.f;
	float tilt = rotation.z * M_PI / 180.f;
	float xDirection = sin(yaw) * cos(pitch);
	float zDirection = sin(pitch);
	float yDirection = cos(yaw) * cos(pitch);

	float3 directionToCamera = float3{ xDirection, yDirection, zDirection };
	float3 viewDirection = directionToCamera * (-1.0f);
	auto v = to_vec3(viewDirection);

	float3 centerPosition{ position.x, position.y, position.z };
	float3 eyePosition = centerPosition;
	auto p = to_vec3(eyePosition);

	auto vert = math::cross(p,v);
        auto u = to_vec3(float3{0, 0, 1});
        if (get<parameters::render_settings::vrtxFlipCameraUp>() == 0) {
          if (fabsf(math::dot(u, v)) > 0.99995f)
            u = to_vec3(float3{0, 1, 0});
        } else if (fabsf(math::dot(u, v)) > 0.99995f)
            u = to_vec3(float3{1, 0, 0});
	auto forward = glm::normalize(v);
	auto up = glm::normalize(u);
	auto strafe = glm::normalize(glm::cross(forward, up));
	p = to_vec3(center);
	bool condition = false;
	auto predicateOf = [&](auto p) {
		auto viewMatrix = glm::lookAt(p,p+v,u);

		auto[x_000, v_000, b_000] = xv(minAABB.x, minAABB.y, minAABB.z, viewMatrix);
		auto[x_001, v_001, b_001] = xv(minAABB.x, minAABB.y, maxAABB.z, viewMatrix);
		auto[x_010, v_010, b_010] = xv(minAABB.x, maxAABB.y, minAABB.z, viewMatrix);
		auto[x_011, v_011, b_011] = xv(minAABB.x, maxAABB.y, maxAABB.z, viewMatrix);

		auto[x_100, v_100, b_100] = xv(maxAABB.x, minAABB.y, minAABB.z, viewMatrix);
		auto[x_101, v_101, b_101] = xv(maxAABB.x, minAABB.y, maxAABB.z, viewMatrix);
		auto[x_110, v_110, b_110] = xv(maxAABB.x, maxAABB.y, minAABB.z, viewMatrix);
		auto[x_111, v_111, b_111] = xv(maxAABB.x, maxAABB.y, maxAABB.z, viewMatrix);

		condition = b_000 && b_001 && b_010 && b_011 && b_100 && b_101 && b_110 && b_111;
		return condition;
	};
	auto updatePositionBackward = [&](auto p, auto stepFactor) {
		int32_t counter = 0;
		do {
			p = p - v * stepFactor;
			condition = predicateOf(p);
		} while (!condition && counter++ < 512);
		return p;
	};
	auto updatePositionForward = [&](auto p, auto stepFactor) {
		int32_t counter = 0;
		do {
			p = p+v * stepFactor;
			condition = predicateOf(p);
		} while (condition && counter++ < 512);
		p = p-v * stepFactor;
		return p;
	};
	p = to_vec3(center);
	//p = updatePositionBackward(p, 4.f);
	//p = updatePositionForward(p, 2.f);
	for (float f = 0.f; f <= 8.f; f += 1.f) {
		p = updatePositionBackward(p, powf(2.f, -f * 2.f));
		p = updatePositionForward(p, powf(2.f, -(f * 2.f + 1.f)));
	}
	position = p;
	updateViewMatrix();
	tracking = true;
}

std::pair<bool, DeviceCamera> Camera::prepareDeviceCamera() {
    // glm::vec3 forward(matrices.view(2, 0), matrices.view(2, 1), matrices.view(2, 2));
    // glm::vec3 strafe(matrices.view(0, 0), matrices.view(1, 0), matrices.view(2, 0));
    // glm::vec3 up(matrices.view(1, 0), matrices.view(1, 1), matrices.view(1, 2));

  DeviceCamera cam;
  float fovx = fov*1.f;
  fovx = get<parameters::render_settings::camera_fov>();
  float2 resolution{(float)width, (float)height};
float2 fov2;
  fov2.x = fovx;
  fov2.y = atan(tan(fovx / 180.f * CUDART_PI_F * 0.5) *
               1.f / aspect) *
          2.0 * 180.f / CUDART_PI_F;
//std::cout << fov2.x << " : " << fov2.y << std::endl;
  cam.resolution = float2{(float)width, (float)height};
  cam.position = float3{position.x, position.y, position.z};
  cam.view = float3{forward.x, forward.y, forward.z};
  cam.up = float3{up.x, up.y, up.z};
  cam.fov = float2{fov2.x, fov2.y};
  static float lastAperture = get<parameters::render_settings::apertureRadius>();
  static float lastFocalDistance = get<parameters::render_settings::focalDistance>();
  static float lastFocalLength = -1.f;
  if (lastAperture != get<parameters::render_settings::apertureRadius>() || lastFocalDistance != get<parameters::render_settings::focalDistance>() || lastFocalLength != get < parameters::render_settings::camera_fov>()) {
	  dirty = true;
	  lastFocalLength = get<parameters::render_settings::camera_fov>();
	  lastAperture = get<parameters::render_settings::apertureRadius>();
	  lastFocalDistance = get<parameters::render_settings::focalDistance>();
  }
  cam.apertureRadius = get<parameters::render_settings::apertureRadius>();
  cam.focalDistance = get<parameters::render_settings::focalDistance>();

  auto MVP = matrices.perspective * matrices.view;

  auto mat_conv = [](auto v_i) {
    Matrix4x4 mat;
    mat(0, 0) = v_i[0][0];
    mat(1, 0) = v_i[1][0];
    mat(2, 0) = v_i[2][0];
    mat(3, 0) = v_i[3][0];

    mat(0, 1) = v_i[0][1];
    mat(1, 1) = v_i[1][1];
    mat(2, 1) = v_i[2][1];
    mat(3, 1) = v_i[3][1];

    mat(0, 2) = v_i[0][2];
    mat(1, 2) = v_i[1][2];
    mat(2, 2) = v_i[2][2];
    mat(3, 2) = v_i[3][2];

    mat(0, 3) = v_i[0][3];
    mat(1, 3) = v_i[1][3];
    mat(2, 3) = v_i[2][3];
    mat(3, 3) = v_i[3][3];
    return mat;
  };

  cam.ViewInverse = mat_conv((glm::inverse(matrices.view)));
  cam.PerspInverse = mat_conv((glm::inverse(matrices.perspective)));
  cam.MVP = mat_conv((glm::inverse(MVP)));
  //std::cout << "View Matrix : \n" <<
  //    cam.ViewInverse(0, 0) << " " << cam.ViewInverse(0, 1) << " " << cam.ViewInverse(0, 2) << " " << cam.ViewInverse(0, 3) << "\n" <<
  //    cam.ViewInverse(1, 0) << " " << cam.ViewInverse(1, 1) << " " << cam.ViewInverse(1, 2) << " " << cam.ViewInverse(1, 3) << "\n" <<
  //    cam.ViewInverse(2, 0) << " " << cam.ViewInverse(2, 1) << " " << cam.ViewInverse(2, 2) << " " << cam.ViewInverse(2, 3) << "\n" <<
  //    cam.ViewInverse(3, 0) << " " << cam.ViewInverse(3, 1) << " " << cam.ViewInverse(3, 2) << " " << cam.ViewInverse(3, 3) << "\n";

  //std::cout << "Perspective Matrix : \n" <<
  //    cam.PerspInverse(0, 0) << " " << cam.PerspInverse(0, 1) << " " << cam.PerspInverse(0, 2) << " " << cam.PerspInverse(0, 3) << "\n" <<
  //    cam.PerspInverse(1, 0) << " " << cam.PerspInverse(1, 1) << " " << cam.PerspInverse(1, 2) << " " << cam.PerspInverse(1, 3) << "\n" <<
  //    cam.PerspInverse(2, 0) << " " << cam.PerspInverse(2, 1) << " " << cam.PerspInverse(2, 2) << " " << cam.PerspInverse(2, 3) << "\n" <<
  //    cam.PerspInverse(3, 0) << " " << cam.PerspInverse(3, 1) << " " << cam.PerspInverse(3, 2) << " " << cam.PerspInverse(3, 3) << "\n";

  //std::cout << "MVP Matrix : \n" <<
  //    cam.MVP(0, 0) << " " << cam.MVP(0, 1) << " " << cam.MVP(0, 2) << " " << cam.MVP(0, 3) << "\n" <<
  //    cam.MVP(1, 0) << " " << cam.MVP(1, 1) << " " << cam.MVP(1, 2) << " " << cam.MVP(1, 3) << "\n" <<
  //    cam.MVP(2, 0) << " " << cam.MVP(2, 1) << " " << cam.MVP(2, 2) << " " << cam.MVP(2, 3) << "\n" <<
  //    cam.MVP(3, 0) << " " << cam.MVP(3, 1) << " " << cam.MVP(3, 2) << " " << cam.MVP(3, 3) << "\n";
  //std::cout << "########################################\n";

  return std::make_pair(dirty, cam);
}

void Camera::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        bool switchPosition = false;
        static glm::vec3 angle{ 0, 0, 0 };
        if (key == GLFW_KEY_F1 || key == GLFW_KEY_F2 || key == GLFW_KEY_F3 ||
            key == GLFW_KEY_F4 || key == GLFW_KEY_F5 || key == GLFW_KEY_F6) {
            switchPosition = true;
        }
        static bool f1 = false, f2 = false, f3 = false, f4 = false, f5 = false, f6 = false;

        switch (key) {
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL:
            angle = glm::vec3{ 0,0,0 };
            f1 = false; f2 = false; f3 = false; f4 = false; f5 = false; f6 = false;
            break;
        case GLFW_KEY_W:
            keys.up = true;
            break;
        case GLFW_KEY_S:
            keys.down = true;
            break;
        case GLFW_KEY_A:
            keys.left = true;
            break;
        case GLFW_KEY_D:
            keys.right = true;
            break;
        case GLFW_KEY_E:
            keys.e = true;
            break;
        case GLFW_KEY_Q:
            keys.q = true;
            break;
        case GLFW_KEY_F1:
            if (! (mods & GLFW_MOD_CONTROL)) {
                setPosition(glm::vec3{ 150, 0, 0 });
                setRotation(glm::vec3{ 180, 0, -90 });
            }
            else f1 = true;
            break;
        case GLFW_KEY_F2:
            if (!(mods & GLFW_MOD_CONTROL)) {
                setPosition(glm::vec3{ 0, 150, 0 });
                setRotation(glm::vec3{ 180, 0, -180 });
            }
            else f2 = true;
            break;
        case GLFW_KEY_F3:
            if (!(mods & GLFW_MOD_CONTROL)) {
                setPosition(glm::vec3{ -150, 0, 0 });
                setRotation(glm::vec3{ 180, 0, 90 });
            }
            else f3 = true;
            break;
        case GLFW_KEY_F4:
            if (!(mods & GLFW_MOD_CONTROL)) {
                setPosition(glm::vec3{ 0, -150, 0 });
                setRotation(glm::vec3{ 180, 0, 0 });
            }
            else f4 = true;
            break;
        case GLFW_KEY_F5:
            if (!(mods & GLFW_MOD_CONTROL)) {
                setPosition(glm::vec3{ 0, 0, -150 });
                setRotation(glm::vec3{ -90, 0, 0 });
            }
            else f5 = true;
            break;
        case GLFW_KEY_F6:
            if (!(mods & GLFW_MOD_CONTROL)) {
                setPosition(glm::vec3{ 0, 0, 150 });
                setRotation(glm::vec3{ 90, 0, 0 });
            }
            else f6 = true;
            break;
        }
        if ((mods & GLFW_MOD_CONTROL)) {
            // 1 key
            if (f1 && !f2 && !f3 && !f4 && !f5 && !f6)
                setRotation(glm::vec3{ 180, 0, 270 });
            if (!f1 && f2 && !f3 && !f4 && !f5 && !f6)
                setRotation(glm::vec3{ 180, 0, 180 });
            if (!f1 && !f2 && f3 && !f4 && !f5 && !f6)
                setRotation(glm::vec3{ 180, 0, 90 });
            if (!f1 && !f2 && !f3 && f4 && !f5 && !f6)
                setRotation(glm::vec3{ 180, 0, 0 });
            if (!f1 && !f2 && !f3 && !f4 && f5 && !f6)
                setRotation(glm::vec3{ -90, 0, 0 });
            if (!f1 && !f2 && !f3 && !f4 && !f5 && f6)
                setRotation(glm::vec3{ 90, 0, 0 });

            // 2 keys
            if (f1 && f2 && !f3 && !f4 && !f5 && !f6)
                setRotation(glm::vec3{ 180, 0, 215 });
            if (!f1 && f2 && f3 && !f4 && !f5 && !f6)
                setRotation(glm::vec3{ 180, 0, 135 });
            if (!f1 && !f2 && f3 && f4 && !f5 && !f6)
                setRotation(glm::vec3{ 180, 0, 45 });
            if (f1 && !f2 && !f3 && f4 && !f5 && !f6)
                setRotation(glm::vec3{ 180, 0, 315 });
            // up 2 key
            if (f1 && !f2 && !f3 && !f4 && f5 && !f6)
                setRotation(glm::vec3{ 225, 0, 270 });
            if (!f1 && f2 && !f3 && !f4 && f5 && !f6)
                setRotation(glm::vec3{ 225, 0, 180 });
            if (!f1 && !f2 && f3 && !f4 && f5 && !f6)
                setRotation(glm::vec3{ 225, 0, 90 });
            if (!f1 && !f2 && !f3 && f4 && f5 && !f6)
                setRotation(glm::vec3{ 225, 0, 0 });
            // down 2 key
            if (f1 && !f2 && !f3 && !f4 && !f5 && f6)
                setRotation(glm::vec3{ 135, 0, -90 });
            if (!f1 && f2 && !f3 && !f4 && !f5 && f6)
                setRotation(glm::vec3{ 135, 0, -180 });
            if (!f1 && !f2 && f3 && !f4 && !f5 && f6)
                setRotation(glm::vec3{ 135, 0, 90 });
            if (!f1 && !f2 && !f3 && f4 && !f5 && f6)
                setRotation(glm::vec3{ 135, 0, 0 });
            // 3 keys up
            if (f1 && f2 && !f3 && !f4 && f5 && !f6)
                setRotation(glm::vec3{ 225, 0, 215 });
            if (!f1 && f2 && f3 && !f4 && f5 && !f6)
                setRotation(glm::vec3{ 225, 0, 135 });
            if (!f1 && !f2 && f3 && f4 && f5 && !f6)
                setRotation(glm::vec3{ 225, 0, 45 });
            if (f1 && !f2 && !f3 && f4 && f5 && !f6)
                setRotation(glm::vec3{ 225, 0, 315 });
            // 3 keys up
            if (f1 && f2 && !f3 && !f4 && !f5 && f6)
                setRotation(glm::vec3{ 135, 0, 215 });
            if (!f1 && f2 && f3 && !f4 && !f5 && f6)
                setRotation(glm::vec3{ 135, 0, 135 });
            if (!f1 && !f2 && f3 && f4 && !f5 && f6)
                setRotation(glm::vec3{ 135, 0, 45 });
            if (f1 && !f2 && !f3 && f4 && !f5 && f6)
                setRotation(glm::vec3{ 135, 0, 315 });
        }
        if (switchPosition && !(mods & GLFW_MOD_SHIFT) ||
            switchPosition && !(mods & GLFW_MOD_CONTROL))
            maximizeCoverage();
        if (switchPosition && (mods & GLFW_MOD_SHIFT))
            tracking = false;
    }
else if(action == GLFW_RELEASE){
switch (key) {
case GLFW_KEY_W:
    keys.up = false;
    break;
case GLFW_KEY_S:
    keys.down = false;
    break;
case GLFW_KEY_A:
    keys.left = false;
    break;
case GLFW_KEY_D:
    keys.right = false;
    break;
case GLFW_KEY_E:
    keys.e = false;
    break;
case GLFW_KEY_Q:
    keys.q = false;
    break;
case GLFW_KEY_L:
    if (mods & GLFW_MOD_SHIFT) {
        get<parameters::render_settings::camera_position>() = float3{ position.x, position.y, position.z };
        get<parameters::render_settings::camera_angle>() = float3{ rotation.x, rotation.y, rotation.z };
    }
    else {
        LOG_DEBUG << "        \"camera_position\": \"" << position.x << " " << position.y << " " << position.z
            << "\"," << std::endl;
        LOG_DEBUG << "        \"camera_angle\": \"" << rotation.x << " " << rotation.y << " " << rotation.z << "\","
            << std::endl;

        float pitch = rotation.x * M_PI / 180.f;
        float yaw = rotation.z * M_PI / 180.f;
        float tilt = rotation.z * M_PI / 180.f;
        //std::cout << yaw << " : " << pitch << std::endl;

        float xDirection = sin(yaw) * cos(pitch);
        float zDirection = sin(pitch);
        float yDirection = cos(yaw) * cos(pitch);
        //std::cout << xDirection << " " << yDirection << " " << zDirection << std::endl;
        float3 centerPosition{ position.x, position.y, position.z };
        float3 directionToCamera = float3{ xDirection, yDirection, zDirection };
        float3 viewDirection = directionToCamera * (-1.0f);
        float3 eyePosition = centerPosition;

        auto to_vec3 = [](auto v) { return glm::vec3(v.x, v.y, v.z); };
        auto p = to_vec3(eyePosition);
        auto v = to_vec3(viewDirection);

        auto vert = glm::cross(p, v);
        //auto u = vert;
        auto u = to_vec3(float3{ 0, 0, 1 });
        if (get<parameters::render_settings::vrtxFlipCameraUp>() == 0) {
            if (fabsf(glm::dot(u, v)) > 0.99995f)
                u = to_vec3(float3{ 0, 1, 0 });
        }
        else if (fabsf(glm::dot(u, v)) > 0.99995f)
            u = to_vec3(float3{ 1, 0, 0 });
        //u = vert;
        forward = glm::normalize(v);
        up = glm::normalize(u);
        strafe = glm::normalize(glm::cross(forward, up));

        matrices.view = glm::lookAt(p, p + v, u);
        std::cout << p << std::endl;
        std::cout << p+v << std::endl;
    }
    break;
case GLFW_KEY_R: {
    auto p = get<parameters::render_settings::camera_position>();
    auto r = get<parameters::render_settings::camera_angle>();
    glm::vec3 pos{ p.x, p.y, p.z };
    glm::vec3 rot{ r.x, r.y, r.z };
    setPosition(pos);
    setRotation(rot);
}

               break;
}

}
}

void Camera::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        mousePos.x = xpos;
        mousePos.y = ypos;
        switch (button) {
        case GLFW_MOUSE_BUTTON_LEFT:
            lbuttondown = true;
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            mbuttondown = true;
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            rbuttondown = true;
            break;
        default:
            break;
        }
        //if (lbuttondown || rbuttondown || mbuttondown) {
        //  QApplication::setOverrideCursor(Qt::BlankCursor);
        //} else {
        //  QApplication::setOverrideCursor(Qt::ArrowCursor);
        //}
    }
    else {
        //setKeyboardModifiers(event);
        switch (button) {
        case GLFW_MOUSE_BUTTON_LEFT:
            lbuttondown = false;
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            mbuttondown = false;
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            rbuttondown = false;
            break;
        default:
            break;
        }
        //if (lbuttondown || rbuttondown || mbuttondown) {
        //    QApplication::setOverrideCursor(Qt::BlankCursor);
        //}
        //else {
        //    QApplication::setOverrideCursor(Qt::ArrowCursor);
        //}
    }
}

void Camera::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
  auto p = glm::vec2(xpos,ypos);
  if (glm::vec2(mousePos.x, mousePos.y) == p)
    return;


  if (lbuttondown) {

    glm::vec3 diff;
    diff.z = (-rotationSpeed * (mousePos.x - (float)xpos));
    diff.x = (rotationSpeed * (mousePos.y - (float)ypos));
    rotate(diff);
    updateViewMatrix();

    glfwSetCursorPos(window, mousePos.x, mousePos.y);
  }
  if (rbuttondown) {
    // glm::vec3 forward(matrices.view(0, 2), matrices.view(1, 2), matrices.view(2, 2));
    // glm::vec3 strafe(matrices.view(0, 0), matrices.view(1, 0), matrices.view(2, 0));
    // glm::vec3 up(matrices.view(0, 1), matrices.view(1, 1), matrices.view(2, 1));

    //glm::vec3 camFront;
    //camFront = forward.normalized();

    position = position + forward * (mousePos.y - (float)ypos) * 0.01f;
    updateViewMatrix();

    glfwSetCursorPos(window, mousePos.x, mousePos.y);
  }
  if (mbuttondown) {
    //glm::vec3 strafe(matrices.view(0, 0), matrices.view(1, 0), matrices.view(2, 0));
    //glm::vec3 up(matrices.view(0, 1), matrices.view(1, 1), matrices.view(2, 1));

    position = position - glm::normalize(strafe) * (mousePos.x - (float)xpos) * 0.01f;
    position = position - glm::normalize(up) * (mousePos.y - (float)ypos) * 0.01f;
    updateViewMatrix();

    glfwSetCursorPos(window, mousePos.x, mousePos.y);
  }
}
void Camera::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {}
float qDegreesToRadians(float deg) {
    return (deg / 360.f) * CUDART_PI_F * 2.f;
}
void Camera::updateViewMatrix() {
  float pitch = rotation.x * M_PI / 180.f;
  float yaw = rotation.z * M_PI / 180.f;
  float tilt = rotation.z * M_PI / 180.f;
  //std::cout << yaw << " : " << pitch << std::endl;

  float xDirection = sin(yaw) * cos(pitch);
  float zDirection = sin(pitch);
  float yDirection = cos(yaw) * cos(pitch);
  //std::cout << xDirection << " " << yDirection << " " << zDirection << std::endl;
  float3 centerPosition{position.x, position.y, position.z};
  float3 directionToCamera = float3{xDirection, yDirection, zDirection};
  float3 viewDirection = directionToCamera * (-1.0f);
  float3 eyePosition = centerPosition;

  auto to_vec3 = [](auto v) { return glm::vec3(v.x, v.y, v.z); };
  auto p = to_vec3(eyePosition);
  auto v = to_vec3(viewDirection);

  auto vert = glm::cross(p, v);
  //auto u = vert;
  auto u = to_vec3(float3{0, 0, 1});
  if (get<parameters::render_settings::vrtxFlipCameraUp>() == 0) {
     if (fabsf(glm::dot(u, v)) > 0.99995f)
      u = to_vec3(float3{0, 1, 0});
  } else if (fabsf(glm::dot(u, v)) > 0.99995f)
    u = to_vec3(float3{1, 0, 0});
  //u = vert;
  forward = glm::normalize(v);
  up = glm::normalize(u);
  strafe = glm::normalize(glm::cross(forward,up));

  matrices.view = glm::lookAt(p,p + v,u);

  glm::mat4 rotM = glm::mat4(1.f);
  glm::mat4  transM = glm::mat4(0.f);
  rotM = glm::rotate(rotM, rotation.x, glm::vec3(1, 0, 0));
  rotM = glm::rotate(rotM, rotation.y, glm::vec3(0, 1, 0));
  rotM = glm::rotate(rotM, rotation.z, glm::vec3(0, 0, 1));
  transM = glm::translate(transM, position);

  if (type == CameraType::firstperson) {
    //matrices.view = rotM * transM;
  } else {
    //matrices.view = transM * rotM;
  }

  // forward = glm::vec3(matrices.view(2, 0), matrices.view(2, 1), matrices.view(2, 2));
  // right = glm::vec3(matrices.view(0, 0), matrices.view(1, 0), matrices.view(2, 0));
  // up = glm::vec3(matrices.view(0, 1), matrices.view(1, 1), matrices.view(2, 1));

glm::mat4 matrix = glm::perspective(qDegreesToRadians(fov), aspect, znear, zfar);
  //std::cout << fov << std::endl;
  matrices.perspective = matrix;

  ParameterManager::instance().get<glm::mat4>("camera.perspective") = matrices.perspective;
  ParameterManager::instance().get<glm::mat4>("camera.view") = matrices.view;
   
  //std::cout << "View Matrix: " << std::endl
  //    << matrices.view[0][0] << " " << matrices.view[0][1] << " " << matrices.view[0][2] << " " << matrices.view[0][3] << "\n"
  //    << matrices.view[1][0] << " " << matrices.view[1][1] << " " << matrices.view[1][2] << " " << matrices.view[1][3] << "\n"
  //    << matrices.view[2][0] << " " << matrices.view[2][1] << " " << matrices.view[2][2] << " " << matrices.view[2][3] << "\n"
  //    << matrices.view[3][0] << " " << matrices.view[3][1] << " " << matrices.view[3][2] << " " << matrices.view[3][3] << "\n";

  //std::cout << "Projection Matrix: " << std::endl
  //    << matrices.perspective[0][0] << " " << matrices.perspective[0][1] << " " << matrices.perspective[0][2] << " " << matrices.perspective[0][3] << "\n"
  //    << matrices.perspective[1][0] << " " << matrices.perspective[1][1] << " " << matrices.perspective[1][2] << " " << matrices.perspective[1][3] << "\n"
  //    << matrices.perspective[2][0] << " " << matrices.perspective[2][1] << " " << matrices.perspective[2][2] << " " << matrices.perspective[2][3] << "\n"
  //    << matrices.perspective[3][0] << " " << matrices.perspective[3][1] << " " << matrices.perspective[3][2] << " " << matrices.perspective[3][3] << "\n";


  dirty = true;
  tracking = false;
}
bool Camera::moving() { return keys.left || keys.right || keys.up || keys.down || keys.e || keys.q; }
void Camera::setPerspective(float fov, float aspect, float znear, float zfar) {
  this->fov = fov;
  this->znear = znear;
  this->zfar = zfar;
  this->aspect = aspect;
  //glViewport(0, 0, width, height);

  glm::mat4 matrix = glm::perspective(qDegreesToRadians(fov), aspect, znear, zfar);
  matrices.perspective = matrix;

  ParameterManager::instance().get<glm::mat4>("camera.perspective") = matrices.perspective;
  ParameterManager::instance().get<glm::mat4>("camera.view") = matrices.view;
  dirty = true;
}
void Camera::updateAspectRatio(float aspect) { setPerspective(fov, aspect, znear, zfar); }
void Camera::setPosition(glm::vec3 position) {
  this->position = position;
  updateViewMatrix();
}
void Camera::setRotation(glm::vec3 rotation) {
  this->rotation = rotation;
  updateViewMatrix();
}
void Camera::rotate(glm::vec3 delta) {
  this->rotation = this->rotation + delta;
  updateViewMatrix();
}
void Camera::setTranslation(glm::vec3 translation) {
  this->position = translation;
  updateViewMatrix();
}
void Camera::translate(glm::vec3 delta) {
  this->position = this->position + delta;
  updateViewMatrix();
}

void Camera::update(float deltaTime) {
	static int32_t f = -1;
	if (tracking && f != get<parameters::internal::frame>()) {
		maximizeCoverage();
		f = get<parameters::internal::frame>();
	}
  if (type == CameraType::firstperson) {
      //std::cout << "Moving: " << moving() << " " <<
      //    (keys.up ? "U" : " ") << (keys.down ? "D" : " ") <<
      //    (keys.left ? "L" : " ") << (keys.right ? "R" : " ") <<
      //    (keys.q ? "Q" : " ") << (keys.e ? "E" : " ") << " -> " <<
      //    (lbuttondown ? "L" : " ") <<
      //    (mbuttondown ? "M" : " ") <<
      //    (rbuttondown ? "R" : " ") << std::endl;
    if (moving()) {
      glm::vec3 camFront;
      camFront.x = (-cos(qDegreesToRadians(rotation.x)) * sin(qDegreesToRadians(rotation.y)));
      camFront.y = (sin(qDegreesToRadians(rotation.x)));
      camFront.z = (cos(qDegreesToRadians(rotation.x)) * cos(qDegreesToRadians(rotation.y)));
      camFront = glm::normalize(camFront);
      
    //   glm::vec3 forward(matrices.view(2, 0), matrices.view(2, 1), matrices.view(2, 2));
    //   glm::vec3 strafe(matrices.view(0, 0), matrices.view(0, 1), matrices.view(0, 2));
    //   glm::vec3 up(matrices.view(1, 0), matrices.view(1, 1), matrices.view(1, 2));

    // glm::vec3 forward(matrices.view(2, 0), matrices.view(2, 1), matrices.view(2, 2));
    // glm::vec3 strafe(matrices.view(0, 0), matrices.view(1, 0), matrices.view(2, 0));
    // glm::vec3 up(matrices.view(1, 0), matrices.view(1, 1), matrices.view(1, 2));

      forward = glm::normalize(forward);
      strafe = glm::normalize(strafe);

      glm::vec3 upl = glm::cross(forward,strafe);
      up = glm::normalize(up);

      if (deltaTime < 0.02f)
        deltaTime = 0.02f;
      float moveSpeed = deltaTime * movementSpeed;

      if (keys.up)
        position = position + (forward)*moveSpeed;
      if (keys.down)
        position = position - (forward)*moveSpeed;
      if (keys.left)
        position = position - (strafe)*moveSpeed;
      if (keys.right)
        position = position + (strafe)*moveSpeed;
      if (keys.e)
        position = position + (up)*moveSpeed;
      if (keys.q)
        position = position - (up)*moveSpeed;

      updateViewMatrix();
    }
  }
}
