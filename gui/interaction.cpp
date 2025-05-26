#include "glui.h"
#include <imgui/imgui.h>
#include <IO/config/config.h>
#include <iostream>
#include <simulation/particleSystem.h>
#include <utility/identifier/uniform.h>
#include <IO/particle/particle.h>
#include <IO/renderData/particle.h>

void GUI::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	static std::string map = get<parameters::color_map::map>();
	static std::string buf = get<parameters::color_map::buffer>();
	static bool direction = get<parameters::color_map::autoScaling>();
	static bool scaling = get<parameters::color_map::visualizeDirection>();
	static bool flipped = get<parameters::color_map::map_flipped>();
	static float min = get<parameters::color_map::min>();
	static float max = get<parameters::color_map::max>();
	static bool vScaling = get<parameters::color_map::vectorScaling>();
	static float vScale = get<parameters::color_map::vectorScale>();
	static std::string vMode = get<parameters::color_map::vectorMode>();
	
  auto set_colormap = [&](float min, float max, bool auto_scale, bool visualizeVectors, bool vectorScaling, float vectorScale, std::string render_buffer, std::string render_colormap, int flipped = 0, std::string vectorMode = std::string("length")) {
	  if (!(mods & GLFW_MOD_ALT)) {
		  get<parameters::color_map::min>() = min;
		  get<parameters::color_map::max>() = max;
		  get<parameters::color_map::buffer>() = render_buffer;
		  get<parameters::color_map::autoScaling>() = auto_scale; 
		  get<parameters::color_map::visualizeDirection>() = visualizeVectors;
		  get<parameters::color_map::vectorScale>() = vectorScale;
		  get<parameters::color_map::vectorScaling>() = vectorScaling;
		  get<parameters::color_map::vectorMode>() = vectorMode;
	  }
	  get<parameters::color_map::map>() = render_colormap;
	  get<parameters::color_map::map_flipped>() = flipped;
  }; 
  if (action != GLFW_PRESS) {
	  Camera::instance().keyCallback(window, key, scancode, action, mods);
	  return;
  }
  switch (key) {
  case GLFW_KEY_H: {
	  m_showText = !m_showText;
  } break;
  case GLFW_KEY_P:
	  if (mods & GLFW_MOD_SHIFT)
		  cuda_particleSystem::instance().single = true;
	  else
		cuda_particleSystem::instance().running = !cuda_particleSystem::instance().running;
    break;
  case GLFW_KEY_G: {
	  std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock);
	  auto cfg = get<parameters::internal::target>();
	  if (cfg == launch_config::device) 
		  get<parameters::internal::target>() = launch_config::host;
	  else if (cfg == launch_config::host)
		  get<parameters::internal::target>() = launch_config::device;
	  break;
  }
  case GLFW_KEY_X: {
    std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock);
    IO::config::take_snapshot();
  } break;
  case GLFW_KEY_Z: {
	  std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock); {
		  IO::config::load_snapshot();
		  m_parameterMappings.clear();
		  parameters::iterateParameters([&](auto& param, std::string name) {
			  using T = std::decay_t<decltype(param)>;
			  if constexpr (math::dimension<T>::value != 0xDEADBEEF)
				  m_parameterMappings[name.substr(name.find(".") + 1)] = new gl_uniform_custom<T>(&param, name.substr(name.find(".") + 1));
			  });
		  for (auto renderer : m_renderFunctions) {
			  glUseProgram(renderer->m_program);
			  glBindVertexArray(renderer->vao);
			  for (auto arr : m_parameterMappings)
				  arr.second->add_uniform(renderer->m_program);
			  glBindVertexArray(0);
			  glUseProgram(0);
		  }
	  }
  } break; 
  case GLFW_KEY_C: {
    std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock);
    IO::config::clear_snapshot();
  } break;
  case GLFW_KEY_J: {
	  IO::renderData::saveParticles();
  }break;
  case GLFW_KEY_U: {
	  get<parameters::internal::dumpForSSSPH>() = 1;
  } break;
  case GLFW_KEY_M: { 
	  int flip = get<parameters::color_map::map_flipped>();
	  if (flip == 1)	get<parameters::color_map::map_flipped>() = 0;
	  else				get<parameters::color_map::map_flipped>() = 1;
} break;
  case GLFW_KEY_V: { 
    for(auto& render : m_volumeRenderFunctions)
      render->toggle();
} break;
  case GLFW_KEY_T: {
	  static float old_min = get<parameters::color_map::min>();
	  static float old_max = get<parameters::color_map::max>();
	  if (get<parameters::color_map::autoScaling>() == 1) {
		  get<parameters::color_map::min>() = old_min;
		  get<parameters::color_map::max>() = old_max;
		  get<parameters::color_map::autoScaling>() = false;
	  }
	  else {
		  old_min = get<parameters::color_map::min>();
		  old_max = get<parameters::color_map::max>();
		  get<parameters::color_map::autoScaling>() = true;
	  }
  }break;
  case GLFW_KEY_1: { set_colormap(0.f, 1.f, true, false, false, 1.f, "densityBuffer", "inferno",1); }break;
  case GLFW_KEY_2: { set_colormap(0.f, 1.f, true, false, false, 1.f, "neighborListLength", "jet",1); }break;
  case GLFW_KEY_3: { set_colormap(0.f, 1.f, true, false, false, 1.f, "MLMResolution", "Dark2",1); }break;
  case GLFW_KEY_4: { set_colormap(0.f, 30.f, false, false, false, 1.f, "velocity", "magma",0); }break;
  case GLFW_KEY_5: { set_colormap(-0.25f, 0.75f, true, false, false, 1.f, "particle_type", "tab20c"); }break;
  case GLFW_KEY_6: { set_colormap(0.f, 1.f, true, false, false, 1.f, "lifetime", "plasma"); }break;
  case GLFW_KEY_7: { set_colormap(0.f, 1.f, true, false, false, 1.f, "volume", "gist_heat",1); }break;
  case GLFW_KEY_8: { set_colormap(0.f, 1.f, true, true, true, 5.f, "debugArray", "magma", 1, "w"); }break;
  case GLFW_KEY_9: { set_colormap(0.f, 1.f, true, false, false, 1.f, "distanceBuffer", "viridis"); }break;
  case GLFW_KEY_0: { set_colormap(-2.f, 2.f, false, false, false, 1.f, "adaptive.classification", "RdBu"); }break;
  case GLFW_KEY_MINUS: { set_colormap(min, max, direction, vScaling, vScale, scaling, buf, map, flipped, vMode); }break;
  case GLFW_KEY_O: { get<parameters::internal::dumpNextframe>() = 1; } break;
  case GLFW_KEY_I: { IO::particle::saveParticles(); } break;
  case GLFW_KEY_F7: { Camera::instance().tracking = !Camera::instance().tracking; } break;
  case GLFW_KEY_F8: { pickerActive = !pickerActive; } break;
  //case GLFW_KEY_F8: {mlmTracer->bValid = !mlmTracer->bValid; vrtxTracer->bValid = false; } break;
  case GLFW_KEY_F9: {vrtxTracer->bValid = !vrtxTracer->bValid; } break;
  case GLFW_KEY_F10: {
	  if (mods = 0x00)
		  get<parameters::color_map::visualizeDirection>() = !get<parameters::color_map::visualizeDirection>();
	  if (mods & GLFW_MOD_ALT)
		  get<parameters::color_map::vectorScale>() = -get<parameters::color_map::vectorScale>();
	  if (mods & GLFW_MOD_SHIFT)
		  get<parameters::color_map::vectorScaling>() = -get<parameters::color_map::vectorScaling>();
  
  } break;
  }
  for (const auto& renderer : m_renderFunctions)
	  renderer->keyCallback(window, key, scancode, action, mods);
  for (const auto& renderer : m_volumeRenderFunctions)
	  renderer->keyCallback(window, key, scancode, action, mods);
  //for (const auto& renderer : m_rayTracingFunctions)
	//  renderer->keyCallback(window, key, scancode, action, mods);
  if (vrtxTracer != nullptr)
	  vrtxTracer->keyCallback(window, key, scancode, action, mods);
  Camera::instance().keyCallback(window, key, scancode, action, mods);
}
void GUI::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	Camera::instance().mouseButtonCallback(window, button, action, mods);
}

//#include <glrender/glparticleIndexRender/particleIndexRender.h>
#include <imgui/imgui_internal.h>
void GUI::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
	auto GImGui = ImGui::GetCurrentContext();
	if(!GImGui->HoveredWindow)
		Camera::instance().cursorPositionCallback(window, xpos, ypos); 

	////bool pickerActive = false;
	////int32_t pickedParticle = -1;
	//if (!pickerActive) return;
	//QPoint globalPos = (rect().topLeft());
	//QPoint globalPosMax = (rect().bottomRight());

	//IndexRenderer* idr = (IndexRenderer*)indexRender;
	//auto p = event->pos();

	//double xpos, ypos;
	//xpos = p.x();// -globalPos.x();
	//ypos = globalPosMax.y() - p.y();// +globalPos.y();
	//pickerX = (int32_t)xpos;
	//pickerY = (int32_t)ypos;
	//std::cout << xpos << " : " << ypos << " -> " << idr->pxlData[(int32_t)xpos + (int32_t)ypos * 1920] << std::endl;
}
void GUI::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) { }
