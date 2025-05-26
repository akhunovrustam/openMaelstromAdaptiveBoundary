#include "glui.h"
void GUI::OSD(){
    if (!m_showText)return;
    const float DISTANCE = 10.0f;
    static int corner = 0;
    ImGuiIO& io = ImGui::GetIO();
    if (corner != -1)
    {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 work_area_pos = viewport->GetWorkPos();   // Instead of using viewport->Pos we use GetWorkPos() to avoid menu bars, if any!
        ImVec2 work_area_size = viewport->GetWorkSize();
        ImVec2 window_pos = ImVec2((corner & 1) ? (work_area_pos.x + work_area_size.x - DISTANCE) : (work_area_pos.x + DISTANCE), (corner & 2) ? (work_area_pos.y + work_area_size.y - DISTANCE) : (work_area_pos.y + DISTANCE));
        ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
        ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
        ImGui::SetNextWindowViewport(viewport->ID);
    }
    ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
    if (ImGui::Begin("Stats overlay", &m_showText, (corner != -1 ? ImGuiWindowFlags_NoMove : 0) | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
    {
        auto addParameter = [&](auto paramName) {
            auto& param = ParameterManager::instance().getParameter(paramName);
            ImGui::PushID(param.identifier.c_str());
            ParameterManager::instance().uiFunctions[param.type](param);
            ImGui::PopID();
        };

        addParameter("simulationTime");
        addParameter("timestep");
        //addParameter("timestep_min");
        //addParameter("timestep_max");
        addParameter("simTime");
        addParameter("renderTime");
        addParameter("num_ptcls");
        if (get<parameters::modules::pressure>() == "DFSPH") {
            addParameter("densitySolverIterations");
            addParameter("densityError");
            addParameter("divergenceSolverIterations");
            addParameter("divergenceError");
        }
        addParameter("color_map.buffer");
        addParameter("color_map.min");
        addParameter("color_map.max");

        if (get<parameters::modules::adaptive>() == true) {
            //addParameter("splitPtcls");
            //addParameter("mergedPtcls");
            //addParameter("sharedPtcls");
            //addParameter("blendedPtcls");
            addParameter("ratio");
        }
        //if (get<parameters::modules::surfaceDistance>() == true) {
        //    sstream << "surface  : " << std::setw(11) << get<parameters::surfaceDistance::surface_iterations>() << std::endl;
        //}
        if (get<parameters::modules::neighborhood>() == "constrained") {
            addParameter("support_current_iteration");
            addParameter("adjusted_particles");
        }
        if (get<parameters::modules::resorting>() == "hashed_cell" || get<parameters::modules::resorting>() == "MLM") {
            addParameter("valid_cells");
            addParameter("collision_cells");
            addParameter("gridSize");
        }
        if (get<parameters::modules::rayTracing>() == 1) {
            addParameter("auxCells");
            addParameter("auxCollisions");
        }
    }
    ImGui::End();
}
