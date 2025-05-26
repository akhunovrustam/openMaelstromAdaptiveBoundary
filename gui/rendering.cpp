#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <glad/glad.h> 
#include "glui.h"
#include <simulation/particleSystem.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <render/util/stb_image_write.h>
#include <thread>

GLuint shader_programme;
GLuint vao = 0;
void GUI::renderFunctions() {
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.2f, 0.2f, 0.2f, 1.f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glFlush();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    if (vrtxTracer != nullptr && vrtxTracer->bValid)
        vrtxTracer->render();
    else
        for (auto renderer : m_renderFunctions)
            renderer->render();

    glBindVertexArray(0);
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
}
void progressBar(int32_t frame, int32_t frameTarget, float progress) {
    std::ios cout_state(nullptr);
    cout_state.copyfmt(std::cout);
    static auto startOverall = std::chrono::high_resolution_clock::now();
    static auto startFrame = startOverall;
    static auto lastTime = startOverall;
    static int32_t lastFrame = frame;
    if (frame != lastFrame) {
        lastFrame = frame;
        startFrame = std::chrono::high_resolution_clock::now();
    }
    auto now = std::chrono::high_resolution_clock::now();
    lastTime = now;
    int barWidth = std::min(get<parameters::render_settings::renderSteps>(), 70);
    std::cout << "Rendering " << std::setw(4) << frame;
    if (frameTarget != -1)
        std::cout << "/" << std::setw(4) << frameTarget;
    std::cout << " [";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << int(progress * 100.0) << " ";
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - startFrame);
    if (dur.count() < 100 || progress < 1e-3f) {
        std::cout << " ---/---s  ";
    }
    else {
        auto totalTime =
            ((float)std::chrono::duration_cast<std::chrono::microseconds>(now - startFrame).count()) / 1000.f / 1000.f;
        std::cout << std::fixed << std::setprecision(0) << " " << std::setw(3) << totalTime << "/" << std::setw(3)
            << (totalTime / progress) << "s  ";
    }
    if (frameTarget != -1 && frame != 0) {
        auto duration = now - startOverall;
        auto progress =
            ((float)(frame * get<parameters::render_settings::camera_fps>())) / ((float)(frameTarget * get<parameters::render_settings::camera_fps>()));
        auto estimated = duration / progress - duration;
        auto printTime = [](auto tp) {
            std::stringstream sstream;
            auto h = std::chrono::duration_cast<std::chrono::hours>(tp).count();
            auto m = std::chrono::duration_cast<std::chrono::minutes>(tp).count() - h * 60;
            auto s = std::chrono::duration_cast<std::chrono::seconds>(tp).count() - h * 3600 - m * 60;
            sstream << std::setw(2) << h << "h " << std::setw(2) << m << "m " << std::setw(2) << s << "s";
            return sstream.str();
        };
        std::cout << " Elapsed: " << printTime(duration) << " ETA: " << printTime(estimated);
        std::cout << "     ";
    }

    std::cout << "\r";
    std::cout.flush();
    std::cout.copyfmt(cout_state);
}
void progressBar2(int32_t frame, int32_t frameTarget) {
    std::ios cout_state(nullptr);
    cout_state.copyfmt(std::cout);
    static auto startOverall = std::chrono::high_resolution_clock::now();
    if (frameTarget == 0) {
        startOverall = std::chrono::high_resolution_clock::now();
        return;
    }
    auto now = std::chrono::high_resolution_clock::now();

    int barWidth = 70;
    std::cout << "Rendering " << std::setw(4) << frame;
    if (frameTarget != -1)
        std::cout << "/" << std::setw(4) << frameTarget;
    std::cout << " [";

    float progress = ((float)frame) / ((float)frameTarget);
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% ";
    if (frame != 0) {
        auto duration = now - startOverall;
        auto estimated = duration / progress - duration;
        auto printTime = [](auto tp) {
            std::stringstream sstream;
            auto h = std::chrono::duration_cast<std::chrono::hours>(tp).count();
            auto m = std::chrono::duration_cast<std::chrono::minutes>(tp).count() - h * 60;
            auto s = std::chrono::duration_cast<std::chrono::seconds>(tp).count() - h * 3600 - m * 60;
            sstream << std::setw(2) << h << "h " << std::setw(2) << m << "m " << std::setw(2) << s << "s";
            return sstream.str();
        };
        std::cout << " Elapsed: " << printTime(duration) << " ETA: " << printTime(estimated);
        std::cout << "     ";
    }
    std::cout << "\r";
    std::cout.flush();
    std::cout.copyfmt(cout_state);
}
#include <IO/renderData/particle.h>
void GUI::renderLoop() {
    auto& cmd_line = arguments::cmd::instance();
    if (cmd_line.end_simulation_frame || cmd_line.end_simulation_time)
        cuda_particleSystem::instance().running = true;
    show_demo_window = true;
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    while (!shouldStop && !glfwWindowShouldClose(window)) {
        static int32_t frame = get<parameters::internal::frame>();
        static float last_time = get<parameters::internal::simulationTime>() - 1.f / get<parameters::render_settings::camera_fps>();
        static float target_time = arguments::cmd::instance().time_limit;
        static float timeToSimulate = target_time - get<parameters::internal::simulationTime>();
        static int32_t timeStepsToSimulate =
            arguments::cmd::instance().end_simulation_time ? floorf(timeToSimulate * get<parameters::render_settings::camera_fps>()) : -1;
        static int32_t frameCtr = 0;
        static int32_t exportedFrames = 0;
        if ((frame != get<parameters::internal::frame>()) &&
            (get<parameters::modules::gl_record>() || arguments::cmd::instance().renderToFile) &&
            (get<parameters::internal::simulationTime>() > last_time + 1.f / get<parameters::render_settings::camera_fps>())) {
            last_time += 1.f / get<parameters::render_settings::camera_fps>();
            cuda_particleSystem::instance().running = false;
            if (arguments::cmd::instance().renderToFile) {
                m_renderData.clear();
                prettyRender = true;
            }
            else
                writeFlag = true;
            frameCtr = 0;
        }
        if (arguments::cmd::instance().renderToFile && vrtxTracer != nullptr) {
            if (!prettyRender)
                vrtxTracer->bValid = false;
            if (prettyRender)
                vrtxTracer->bValid = true;
        }
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock);
        cuda_particleSystem::instance().renderFlag = true;
        Camera::instance().update(1.f / ImGui::GetIO().Framerate);
        auto& hScene = hostScene();
        Camera::instance().width = get<parameters::render_settings::camera_resolution>().x;
        Camera::instance().height = get<parameters::render_settings::camera_resolution>().y;
        hScene.width = get<parameters::render_settings::camera_resolution>().x;
        hScene.height = get<parameters::render_settings::camera_resolution>().y;
        auto [dirty, cam] = Camera::instance().prepareDeviceCamera();
        hScene.m_camera = cam;
        hostScene().dirty = dirty;
        Camera::instance().dirty = false;

        ParameterManager::instance().get<float>("renderTime") = 1000.f / ImGui::GetIO().Framerate;
          for (auto arr : m_arrayMappings)
            arr.second->update();
          for (auto uni : m_uniformMappings)
            uni.second->update();
          for (auto arr : m_parameterMappings)
              arr.second->update();

          for (auto renderer : m_renderFunctions)
              renderer->update();
          if (vrtxTracer != nullptr)
              vrtxTracer->update();
        renderFunctions();

        ImGui::NewFrame();
        //ImGui::ShowDemoWindow(&show_demo_window);
        uiFunctions();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }
        glfwSwapBuffers(window);

        if (writeFlag) {
            stbi_flip_vertically_on_write(1);
            if (get<parameters::modules::gl_record>()) {
                
                std::stringstream sstream2;
                sstream2 << "frame_" << std::setfill('0') << std::setw(3) << exportedFrames++ << ".dump";
                get<parameters::simulation_settings::dumpFile>() = (arguments::cmd::instance().renderDirectory / sstream2.str()).string();

                if (exportedFrames % 30 == 0)
                    get<parameters::internal::dumpNextframe>() = 1;

                static int32_t* buffer = new int32_t[Camera::instance().width * Camera::instance().height];
                glReadPixels(0, 0, Camera::instance().width, Camera::instance().height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
//#ifdef WIN32
                fwrite(buffer, sizeof(int) * Camera::instance().width * Camera::instance().height, 1, m_ffmpegPipe);
//#endif
                writeFlag = false;
                prettyRender = false;
                cuda_particleSystem::instance().running = true;
            }
            if (arguments::cmd::instance().renderToFile) {
                static int32_t* buffer = new int32_t[Camera::instance().width * Camera::instance().height];
                glReadPixels(0, 0, Camera::instance().width, Camera::instance().height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
                std::stringstream sstream;
                std::stringstream sstream2;
                sstream << "frame_" << std::setfill('0') << std::setw(3) << exportedFrames++ << ".png";
                sstream2 << "frame_" << std::setfill('0') << std::setw(3) << exportedFrames << ".dump";

                auto fs = arguments::cmd::instance().renderDirectory;
                fs /= sstream.str();
                stbi_write_png(fs.string().c_str(),
                    Camera::instance().width, Camera::instance().height,
                    4, buffer, sizeof(int32_t) * Camera::instance().width);
                writeFlag = false;
                prettyRender = false;
                get<parameters::simulation_settings::dumpFile>() = (arguments::cmd::instance().renderDirectory / sstream2.str()).string();

                if (exportedFrames % 30 == 0)
                    get<parameters::internal::dumpNextframe>() = 1;
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
                cuda_particleSystem::instance().running = true;
            }
        }
        if (prettyRender && arguments::cmd::instance().renderToFile)
            progressBar(exportedFrames, timeStepsToSimulate, (float)frameCtr / (float)get<parameters::render_settings::renderSteps>());
        if (get<parameters::modules::gl_record>())
            progressBar2(exportedFrames, timeStepsToSimulate);
        if (prettyRender && ++frameCtr > get<parameters::render_settings::renderSteps>()) {
            writeFlag = true;
        }

    }
    if (m_ffmpegPipe != nullptr)
#ifdef WIN32
        _pclose(m_ffmpegPipe);
#else
        pclose(m_ffmpegPipe);
#endif
    quit();
}
