#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <glad/glad.h> 
#include "glui.h"
#include <tools/log.h>
#include <config/config.h> 
#include <render/axisRender/axes.h>
#include <render/floorRender/floor.h>
#include <render/boundaryRender/bounds.h>
#include <render/rigidRender/rigid_render.h>
#include <render/particleRender/particle_render.h>
#include <render/vrtxRender/vrtxRender.h>
#include <tools/pathfinder.h>
#include <utility/helpers/arguments.h>
#include <yaml-cpp/yaml.h>
#include <math/template/tuple_for_each.h>
#include <simulation/particleSystem.h> 
#include <gvdb.h>

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}
void GUI::initGL() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        throw std::runtime_error("Could not setup glfw context");
    const char* glsl_version = "#version 450";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    window = glfwCreateWindow(1920, 1080, "Maelstrom - Development Branch", NULL, NULL);
    if (window == NULL)
        throw std::runtime_error("Could not setup window");
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    bool err = gladLoadGL() == 0;
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    if (err) {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        throw std::runtime_error("Failed to initialize OpenGL loader");
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    ImGui::StyleColorsLight();
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    io.Fonts->AddFontFromFileTTF(resolveFile("cfg/Cascadia.ttf").string().c_str(), 16.0f);
    io.Fonts->AddFontFromFileTTF(resolveFile("cfg/NotoMono-Regular.ttf").string().c_str(), 16.0f);
    clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    LOG_DEBUG << "Debug Test";
    LOG_INFO << "Info Test";
    LOG_ERROR << "Error Test";
    //LoggingWindow::instance().AddLog("Starting log...");

    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {GUI::instance().keyCallback(window, key, scancode, action, mods); });
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double xpos, double ypos) {GUI::instance().cursorPositionCallback(window, xpos, ypos); });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {GUI::instance().mouseButtonCallback(window, button, action, mods); });
    glfwSetScrollCallback(window, [](GLFWwindow* window, double xpos, double ypos) {GUI::instance().scrollCallback(window, xpos, ypos); });
    glfwSwapInterval(0);
      for_each(arrays_list, [&](auto x) {
        using T = decltype(x);
        if constexpr (math::dimension<typename T::type>::value != 0xDEADBEEF)
          m_arrayMappings[T::identifier] = new cuda_buffer<T>();
      });
      m_uniformMappings["view_matrix"] =
          new gl_uniform_custom<glm::mat4>(&Camera::instance().matrices.view, "view_matrix");
      m_uniformMappings["perspective_matrix"] =
          new gl_uniform_custom<glm::mat4>(&Camera::instance().matrices.perspective, "perspective_matrix");
      parameters::iterateParameters([&](auto& param, std::string name) {
          using T = std::decay_t<decltype(param)>;
             if constexpr (math::dimension<T>::value != 0xDEADBEEF)
                 m_parameterMappings[name.substr(name.find(".")+1)] = new gl_uniform_custom<T>(&param, name.substr(name.find(".") + 1));
          });
      Camera::instance().type = Camera::CameraType::firstperson;
      Camera::instance().movementSpeed = 25.0f;
      auto p = get<parameters::render_settings::camera_position>();
      auto r = get<parameters::render_settings::camera_angle>();
      Camera::instance().position = glm::vec3(p.x, p.y, p.z);
      Camera::instance().setRotation(glm::vec3(r.x, r.y, r.z));
      Camera::instance().setPerspective(60.0f, (float)1920 / (float)1080, 0.1f, 64000.0f);
      Camera::instance().updateViewMatrix();
    
      if(get<parameters::modules::rayTracing>())
      vrtxTracer = new vRTXrender();

    m_renderFunctions.push_back(new ParticleRenderer());
    m_renderFunctions.push_back(new AxesRenderer());
    m_renderFunctions.push_back(new BoundsRenderer());
    m_renderFunctions.push_back(new FloorRenderer());
      int32_t i = 0;
      for (auto fluidVolume : get<parameters::boundary_volumes::volume>()) {
        auto render = new rigidRender(i++);
        render->toggle();
        m_renderFunctions.push_back(render);
      }

    for (auto renderer : m_renderFunctions) {
        glUseProgram(renderer->m_program);
        glBindVertexArray(renderer->vao);
        for (auto arr : m_arrayMappings)
            arr.second->bindProgram(renderer->m_program);
        for (auto arr : m_uniformMappings)
            arr.second->add_uniform(renderer->m_program);
        for (auto arr : m_parameterMappings)
            arr.second->add_uniform(renderer->m_program);
        glBindVertexArray(0);
        glUseProgram(0);
    }
    if (get<parameters::modules::gl_record>()) {
        std::stringstream sstream;
        sstream << "ffmpeg -r " << get<parameters::render_settings::camera_fps>()
            //<< (!arguments::cmd::instance().vm.count("verbose") ? " -hide_banner -nostats -loglevel 0" : "")
            // << " -hide_banner -nostats -loglevel 0"
            << " -f rawvideo -pix_fmt rgba -s 1920x1080 -i - "
            "-threads 0 -preset ultrafast -y -vf vflip -c:v libx264 "
            "-pix_fmt yuv420p -b:v 50M "
            <</* get<parameters::internal::config_folder>() << */get<parameters::render_settings::gl_file>();
        std::cout << "FFFFFFFFFFFFFFFFFFF " << get<parameters::internal::config_folder>() << get<parameters::render_settings::gl_file>() << "\n";
            #ifdef WIN32
        m_ffmpegPipe = _popen(sstream.str().c_str(), "wb");
        #else
        m_ffmpegPipe = popen(sstream.str().c_str(), "w");

        #endif
    }
    }

void GUI::initSimulation() {
    cuda_particleSystem::instance().init_simulation();
    cuda_particleSystem::instance().running = false;
    cuda_particleSystem::instance().step();

}

std::filesystem::path resolveFileLocal(std::string fileName, std::vector<std::string> search_paths) {
    namespace fs = std::filesystem;
    fs::path expanded = expand(fs::path(fileName));

    fs::path base_path = "";
    if (fs::exists(expand(fs::path(fileName))))
        return expand(fs::path(fileName));
    for (const auto& path : search_paths) {
        auto p = expand(fs::path(path));
        if (fs::exists(p / fileName))
            return p.string() + std::string("/") + fileName;
    }

    if (fs::exists(fileName)) return fs::path(fileName);
    if (fs::exists(expanded))
        return expanded;

    for (const auto& pathi : search_paths) {
        auto path = expand(fs::path(pathi));
    }

    std::stringstream sstream;
    sstream << "File '" << fileName << "' could not be found in any provided search path" << std::endl;
    LOG_ERROR << sstream.str();
    throw std::runtime_error(sstream.str().c_str());
}

void GUI::initParameters(int argc, char* argv[]) {
    auto& pm = ParameterManager::instance();
    std::string stylePath;
    std::string fileName = "cfg/style.css";
    namespace fs = std::filesystem;
    fs::path working_dir = fs::absolute(fs::path(argv[0])).remove_filename();
    fs::path binary_dir = fs::current_path();
    fs::path source_dir = sourceDirectory;
    fs::path build_dir = binaryDirectory;
    fs::path expanded = expand(fs::path(fileName));
    stylePath = resolveFileLocal(fileName, { working_dir.string(),binary_dir.string(),source_dir.string(),build_dir.string() }).string();
    pm.newParameter("stylePath", stylePath, { .hidden = true });
    pm.init();
    auto binary_directory = fs::absolute(fs::path(argv[0])).remove_filename();
    auto working_directory = fs::current_path();

    get<parameters::internal::working_directory>() = working_directory.string();
    get<parameters::internal::binary_directory>() = binary_directory.string();
    get<parameters::internal::source_directory>() = sourceDirectory;
    get<parameters::internal::build_directory>() = binaryDirectory;

    // Initialize the simulation based on command line parameters.
    auto& cmd_line = arguments::cmd::instance();
    if (!cmd_line.init(false, argc, argv))
        throw std::runtime_error("Invalid command line parameters");
}


void GUI::initGVDB() {
    gvdbHelper::initialize();
}