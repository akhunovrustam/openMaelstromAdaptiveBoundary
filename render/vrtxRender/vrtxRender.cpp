#define QT_NO_KEYWORDS
// clang-format off
#include <render/vrtxRender/vrtxRender.h>
#include <simulation/particleSystem.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <render/util/camera.h>
#include <bitset>
#include <iomanip>
#include <render/util/renderer.h>
// clang-format on

vrtx::Refl_t intToMaterial(int32_t v) {
  switch (v) {
  case 0:
    return vrtx::Refl_t::Lambertian;
  case 1:
    return vrtx::Refl_t::Nayar;
  case 2:
    return vrtx::Refl_t::Mirror;
  case 3:
    return vrtx::Refl_t::Plastic;
  case 4:
    return vrtx::Refl_t::Glass;
  default:
    return vrtx::Refl_t::Nayar;
  }
}
std::string materialToString(vrtx::Refl_t v) {
  switch (v) {
  case vrtx::Refl_t::Lambertian:
    return "diffuse";
  case vrtx::Refl_t::Nayar:
    return "coated";
  case vrtx::Refl_t::Mirror:
    return "metallic";
  case vrtx::Refl_t::Plastic:
    return "specular";
  case vrtx::Refl_t::Glass:
    return "refractive";
  default:
    return "diffuse";
  }
}

std::map<std::string, std::variant<int32_t, float, std::vector<int32_t>, std::vector<float>>> vRTXrender::getData() {
  std::map<std::string, std::variant<int32_t, float, std::vector<int32_t>, std::vector<float>>> data;
  for (auto &[timer, times] : timings) {
    if (!std::all_of(times.begin(), times.end(), [](auto v) { return v > 0; })) {
      if (times.size() == 1)
        data[timer] = (int32_t)times[0];
      else {
        auto d = std::vector<int32_t>(times.begin(), times.end()-1);
        std::transform(d.cbegin(), d.cend(), d.begin(), std::negate<int32_t>());
        data[timer] = d;
      }
    } else {
      if (times.size() == 1)
        data[timer] = (float)times[0];
      else
        data[timer] = std::vector<float>(times.begin(), times.end()-1);
    }
  }
  return data;
}
#include <imgui/imgui.h>
void vRTXrender::uiFunction() {
    bool valid = get<parameters::render_settings::vrtxDisplayStats>();
    if (!valid)return;
    // FIXME-VIEWPORT: Select a default viewport
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
    if (ImGui::Begin("Raytracing Overlay", &valid, (corner != -1 ? ImGuiWindowFlags_NoMove : 0) | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
    {
        auto addParameter = [&](auto paramName) {
            auto& param = ParameterManager::instance().getParameter(paramName);
            ImGui::PushID(param.identifier.c_str());
            ParameterManager::instance().uiFunctions[param.type](param);
            ImGui::PopID();
        };
    }
  //if (!valid() || !get<parameters::render_settings::vrtxDisplayStats>())
  //  return "";
  //std::stringstream sstream;
  //// switch (renderMode) {
  //// case 0: sstream << std::string("vRTX [") + std::to_string(framenumber) + "]\n"; break;
  //// case 1: sstream << std::string("vRTX normal debug [") + std::to_string(framenumber) + "]\n"; break;
  //// case 2: sstream << std::string("vRTX transparent [") + std::to_string(framenumber) + "]\n"; break;
  ////}
  //sstream << "Bounce limit   " << get<parameters::render_settings::vrtxBounces>() << "\n";
  //sstream << "Fluid material " << materialToString(intToMaterial(get<parameters::render_settings::vrtxMaterial>())) << "\n";
  //sstream << "Render Fluid   " << (get<parameters::render_settings::vrtxRenderFluid>() ? "yes" : "no") << "\n";
  //sstream << "Render Normals " << (get<parameters::render_settings::vrtxRenderNormals>() ? "yes" : "no") << "\n";
  //sstream << "Surface        " << (get<parameters::render_settings::vrtxRenderSurface>() ? "yes" : "no") << "\n";
  //for (auto &[timer, times] : timings) {
  //  if (std::all_of(times.begin(), times.end(), [](auto v) { return v > 0; })) {
  //    sstream << timer.substr(2) << " ";
  //    for (auto t : times)
  //      sstream << std::setw(8) << std::fixed << std::setprecision(2) << t << "\t";
  //  } else {
  //    sstream << timer.substr(2) << " ";
  //    for (auto t : times)
  //      sstream << std::setw(8) << std::fixed << (int32_t)-t << "\t";
  //  }
  //  sstream << "\n";
  //}
  //return sstream.str();
}
void vRTXrender::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  switch (key) {
  case GLFW_KEY_F11: {
    if (mods & GLFW_MOD_CONTROL)
      get<parameters::render_settings::vrtxRenderFluid>() = !get<parameters::render_settings::vrtxRenderFluid>();
    else if (mods & GLFW_MOD_SHIFT)
      get<parameters::render_settings::vrtxRenderSurface>() = !get<parameters::render_settings::vrtxRenderSurface>();
    else
      get<parameters::render_settings::vrtxRenderNormals>() = !get<parameters::render_settings::vrtxRenderNormals>();
    dirty = true;
    break;
  }
  case GLFW_KEY_F10: {
    if (mods & GLFW_MOD_CONTROL)
      fluidMaterial = fluidMaterial == vrtx::Refl_t::Glass ? vrtx::Refl_t::Plastic : vrtx::Refl_t::Glass;
    else if (mods & GLFW_MOD_SHIFT)
      bounces = std::max(1, bounces--);
    else
      bounces++;
    dirty = true;
    break;
  }
  }
}
#include <tools/pathfinder.h>
#include <render/util/stb_image.h>

void vRTXrender::updateRTX() {
  static std::string tMode = "";
  static std::string mMode = "";
  if (tMode != get<parameters::color_map::transfer_mode>()) {
    if (get<parameters::color_map::transfer_mode>().find("linear") != std::string::npos)
      get<parameters::color_map::transfer_fn>() = 0;
    if (get<parameters::color_map::transfer_mode>().find("cubic") != std::string::npos)
      get<parameters::color_map::transfer_fn>() = 4;
    if (get<parameters::color_map::transfer_mode>().find("cubicRoot") != std::string::npos)
      get<parameters::color_map::transfer_fn>() = 3;
    if (get<parameters::color_map::transfer_mode>().find("square") != std::string::npos)
      get<parameters::color_map::transfer_fn>() = 2;
    if (get<parameters::color_map::transfer_mode>().find("squareRoot") != std::string::npos)
      get<parameters::color_map::transfer_fn>() = 1;
    if (get<parameters::color_map::transfer_mode>().find("log") != std::string::npos)
      get<parameters::color_map::transfer_fn>() = 5;
    tMode = get<parameters::color_map::transfer_mode>();
  }
  if (mMode != get<parameters::color_map::mapping_mode>()) {
    if (get<parameters::color_map::mapping_mode>().find("linear") != std::string::npos)
      get<parameters::color_map::mapping_fn>() = 0;
    if (get<parameters::color_map::mapping_mode>().find("cubic") != std::string::npos)
      get<parameters::color_map::mapping_fn>() = 4;
    if (get<parameters::color_map::mapping_mode>().find("cubicRoot") != std::string::npos)
      get<parameters::color_map::mapping_fn>() = 3;
    if (get<parameters::color_map::mapping_mode>().find("square") != std::string::npos)
      get<parameters::color_map::mapping_fn>() = 2;
    if (get<parameters::color_map::mapping_mode>().find("squareRoot") != std::string::npos)
      get<parameters::color_map::mapping_fn>() = 1;
    if (get<parameters::color_map::mapping_mode>().find("log") != std::string::npos)
      get<parameters::color_map::mapping_fn>() = 5;
    mMode = get<parameters::color_map::mapping_mode>();
  }

  static std::string old_colormap = "";
  static size_t old_size = 0;
  if (old_colormap != get<parameters::color_map::map>()) {
      int32_t image_width = 1024;
      int32_t image_height = 1024;
      float4* img = new float4[1024];

    old_colormap = get<parameters::color_map::map>();
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
    color_map = (float4*)realloc(color_map, sizeof(float4) * (image_width));
    for (int32_t it = 0; it < image_width; ++it) {
        color_map[it] = float4{ (float)(img[it].x) / 256.f, (float)(img[it].y) / 256.f,
                                  (float)(img[it].z) / 256.f, 1.f };
        //std::cout << color_map[it].x() << " : " << color_map[it].y() << " : " << color_map[it].z() << std::endl;
          //if(it == img.width()-1)
          //	color_map[it + 1] = QVector4D{ (float)(col.red()) / 256.f, (float)(col.green()) / 256.f,
          //	(float)(col.blue()) / 256.f, 1.f };
    }

    color_map_elements = image_width;
    delete[] img;
    if (cu_color_map != nullptr)
      cudaFree(cu_color_map);
    cudaMalloc(&cu_color_map, sizeof(float4) * color_map_elements);
    cudaMemcpy(cu_color_map, color_map, sizeof(float4) * color_map_elements, cudaMemcpyHostToDevice);
    dirty = true;
  }
  static int32_t lastSurfacing = -1;
  static int32_t lastFluid = -1;
  static int32_t lastNormal = -1;
  static int32_t lastBVH = -1;
  static int32_t lastMaterial = -1;
  static int32_t lastBVHMaterial = -1;
  static int32_t lastBounces = -1;
  static float lastIOR = 0.f;
  static float3 lastFluidColor = float3{0.f, 0.f, 0.f};
  static float3 lastDebeerColor = float3{0.f, 0.f, 0.f};
  static float lastDebeerScale = 0.f;
  static float lastBias = -1.f;
  static int32_t lastSurfaceExtraction = -1;
  ;
  static int32_t lastRenderGrid = -1;
  static int32_t lastFlipped = -1;
  static int32_t oldTransferFn = -1;
  static int32_t oldMapFn = -1;
  static float old_min = -1.f;
  static float old_max = -1.f;
  static float lastR = -1.f;
  static float lastWmin = -1.f;
  static float lastWmax = -1.f;
  static float lastanisotropicKs = -1.f;
  static int32_t lastDepth = -1;
  static float lastDepthScale = -1.f;
  static float lastEpsilon = -1.f;
  static float3 lastbvhColor{-1.f, 1.f, 2.f};
  static float3 lastvrtxRenderDomainMin{-1.f, -1.f, -1.f};
  static float3 lastvrtxRenderDomainMax{-1.f, -1.f, -1.f};
  if (lastSurfacing != get<parameters::render_settings::vrtxRenderSurface>() || lastFluid != get<parameters::render_settings::vrtxRenderFluid>() ||
      lastNormal != get<parameters::render_settings::vrtxRenderNormals>() || lastMaterial != get<parameters::render_settings::vrtxMaterial>() ||
      lastBounces != get<parameters::render_settings::vrtxBounces>() || lastBVH != get<parameters::render_settings::vrtxRenderBVH>() ||
      lastIOR != get<parameters::render_settings::vrtxIOR>() || lastFluidColor.x != get<parameters::render_settings::vrtxFluidColor>().x ||
      lastFluidColor.y != get<parameters::render_settings::vrtxFluidColor>().y ||
      lastFluidColor.z != get<parameters::render_settings::vrtxFluidColor>().z || lastDebeerColor.x != get<parameters::render_settings::vrtxDebeer>().x ||
      lastDebeerColor.y != get<parameters::render_settings::vrtxDebeer>().y || lastDebeerColor.z != get<parameters::render_settings::vrtxDebeer>().z ||
      lastDebeerScale != get<parameters::render_settings::vrtxDebeerScale>() || lastBVHMaterial != get<parameters::render_settings::vrtxBVHMaterial>() ||
      lastBias != get<parameters::render_settings::vrtxFluidBias>() ||
      lastSurfaceExtraction != get<parameters::render_settings::vrtxSurfaceExtraction>() ||
      lastRenderGrid != get<parameters::render_settings::vrtxRenderGrid>() || lastFlipped != get<parameters::color_map::map_flipped>() ||
      old_min != get<parameters::color_map::min>() || old_max != get<parameters::color_map::max>() ||
      oldTransferFn != get<parameters::color_map::transfer_fn>() ||
      oldMapFn != get<parameters::color_map::mapping_fn>() || lastR != get<parameters::render_settings::vrtxR>() ||
      lastWmin != get<parameters::render_settings::vrtxWMin>() || lastDepth != get<parameters::render_settings::vrtxDepth>() ||
      lastDepthScale != get<parameters::render_settings::vrtxDepthScale>() || lastEpsilon != get<parameters::render_settings::vrtxDomainEpsilon>() ||
          lastbvhColor != get<parameters::render_settings::bvhColor>() ||
      lastvrtxRenderDomainMin != get<parameters::render_settings::vrtxRenderDomainMax>() || lastvrtxRenderDomainMax != get<parameters::render_settings::vrtxRenderDomainMin>()||
      lastWmin != get<parameters::render_settings::vrtxWMin>()|| lastWmax != get<parameters::render_settings::vrtxWMax>()) {
    lastSurfacing = get<parameters::render_settings::vrtxRenderSurface>();
    lastFluid = get<parameters::render_settings::vrtxRenderFluid>();
    lastNormal = get<parameters::render_settings::vrtxRenderNormals>();
    lastMaterial = get<parameters::render_settings::vrtxMaterial>();
    lastBounces = get<parameters::render_settings::vrtxBounces>();
    lastIOR = get<parameters::render_settings::vrtxIOR>();
    lastBVH = get<parameters::render_settings::vrtxRenderBVH>();
    lastFluidColor = get<parameters::render_settings::vrtxFluidColor>();
    lastDebeerColor = get<parameters::render_settings::vrtxDebeer>();
    lastDebeerScale = get<parameters::render_settings::vrtxDebeerScale>();
    lastBVHMaterial = get<parameters::render_settings::vrtxBVHMaterial>();
    lastBias = get<parameters::render_settings::vrtxFluidBias>();
    lastSurfaceExtraction = get<parameters::render_settings::vrtxSurfaceExtraction>();
    lastRenderGrid = get<parameters::render_settings::vrtxRenderGrid>();
    lastFlipped = get<parameters::color_map::map_flipped>();
    old_min = get<parameters::color_map::min>();
    old_max = get<parameters::color_map::max>();
    oldTransferFn = get<parameters::color_map::transfer_fn>();
    oldMapFn = get<parameters::color_map::mapping_fn>();
    lastR = get<parameters::render_settings::vrtxR>();
    lastWmin = get<parameters::render_settings::vrtxWMin>();
    lastWmax = get<parameters::render_settings::vrtxWMax>();
    lastanisotropicKs = get<parameters::render_settings::anisotropicKs>();
    lastDepth = get<parameters::render_settings::vrtxDepth>();
    lastDepthScale = get<parameters::render_settings::vrtxDepthScale>();
    lastEpsilon = get<parameters::render_settings::vrtxDomainEpsilon>();
    lastbvhColor = get<parameters::render_settings::bvhColor>();
    lastvrtxRenderDomainMin = get<parameters::render_settings::vrtxRenderDomainMax>();
    lastvrtxRenderDomainMax = get<parameters::render_settings::vrtxRenderDomainMin>();
    // if (lastRenderGrid) lastFluid = 0;
    dirty = true;
  }
  loadBoxes();
  loadSpheres();
}
void vRTXrender::loadSpheres() {
  if (get<parameters::rtxScene::sphere>().size() == 0) {
    spheres = std::vector<vrtx::Sphere>{
        // vrtx::Sphere{16, {192.0f, 192, 192}, {1.f, 1.f, 1.f}, {0.f, 0.f, 0.f}, vrtx::DIFF},
        vrtx::Sphere{32, {-96, 0, 16}, {0, 0, 0}, {1.f, 1.f, 1.f}, vrtx::Lambertian},
        vrtx::Sphere{32, {-96, -64, 16}, {0, 0, 0}, {0.5f, 0.f, 0.f}, vrtx::Lambertian},
        vrtx::Sphere{32, {-96, 64, 64}, {0, 0, 0}, {1.0f, 1.f, 1.f}, vrtx::Lambertian},
        vrtx::Sphere{10000, {50.0f, 40.8f, -1060}, {0.55f, 0.55f, 0.55f}, {0.075f, 0.075f, 0.075f}, vrtx::Lambertian},
        // vrtx::Sphere{10000, {50.0f, 40.8f, -1060}, {0.55, 0.55, 0.55}, {0.175f, 0.175f, 0.175f}, vrtx::DIFF},
        // vrtx::Sphere{10000, {50.0f, 40.8f, -1060}, {0.f,0.f,0.f}, {0.f,0.f,0.f}, vrtx::DIFF},

        vrtx::Sphere{100000, {0.0f, 0, -100000.f}, {0, 0, 0}, {0.2f, 0.2f, 0.2f}, vrtx::Lambertian},
        vrtx::Sphere{100000, {0.0f, 0, -100000.1f}, {0, 0, 0}, {0.3f, 0.3f, 0.3f}, vrtx::Lambertian}};
    return;
  }
  auto eq = [](auto lhs, auto rhs) { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; };
  if (spheres.size() != get<parameters::rtxScene::sphere>().size())
    spheres.resize(get<parameters::rtxScene::sphere>().size());
  int32_t i = 0;
  for (const auto &sphere : get<parameters::rtxScene::sphere>()) {
    float3 color = sphere.color;
    float3 emission = sphere.emission;
    float3 position = sphere.position;
    float radius = sphere.radius;
    vrtx::Refl_t material = intToMaterial(sphere.refl_t);
    if (!eq(color, spheres[i].col) || !eq(emission, spheres[i].emi) || !eq(position, spheres[i].pos) ||
        radius != spheres[i].rad || material != spheres[i].refl) {
      spheres[i].col = color;
      spheres[i].emi = emission;
      spheres[i].pos = position;
      spheres[i].rad = radius;
      spheres[i].refl = material;
      boxesDirty = true;
      dirty = true;
    }
    i++;
  }
}
#include <sstream>

void vRTXrender::loadBoxes() {
  if (get<parameters::rtxScene::box>().size() == 0) {
    boxes = std::vector<vrtx::Box>{
        // vrtx::Box{{-25.f, -25.f, 96.f},{25.f,25.f, 132.f},{1.f,1.f,1.f}, {0.f,0.f,0.f}, vrtx::DIFF},
        vrtx::Box{{190.f, -192.f, -192.f}, {192.f, 192.f, 192.f}, {1.f, 1.f, 1.f}, {0.f, 0.f, 0.f}, vrtx::Lambertian},
        vrtx::Box{
            {-521, -FLT_MAX, -FLT_MAX}, {-51, FLT_MAX, FLT_MAX}, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, vrtx::Lambertian}
        //,vrtx::Box{ {-FLT_MAX, -25.f, -FLT_MAX},{32, FLT_MAX, FLT_MAX},{0.f,0.f,0.f}, {1.f, 1.f, 1.f}, vrtx::DIFF}
    };
    return;
  }
  auto strToFloat3 = [](auto str) {
    std::istringstream iss(str);
    std::string a, b, c;
    iss >> a >> b >> c;
    float x = (a == "FLT_MAX" ? FLT_MAX : (a == "-FLT_MAX" ? -FLT_MAX : std::stof(a)));
    float y = (b == "FLT_MAX" ? FLT_MAX : (b == "-FLT_MAX" ? -FLT_MAX : std::stof(b)));
    float z = (c == "FLT_MAX" ? FLT_MAX : (c == "-FLT_MAX" ? -FLT_MAX : std::stof(c)));
    return float3{x, y, z};
  };
  auto eq = [](auto lhs, auto rhs) { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; };
  if (boxes.size() != get<parameters::rtxScene::box>().size() + get<parameters::moving_plane::plane>().size())
    boxes.resize(get<parameters::rtxScene::box>().size() + get<parameters::moving_plane::plane>().size());
  int32_t i = 0;
  for (const auto &sphere : get<parameters::rtxScene::box>()) {
    float3 color = sphere.color;
    float3 emission = sphere.emission;
    float3 minPosition = strToFloat3(sphere.minPosition);
    float3 maxPosition = strToFloat3(sphere.maxPosition);
    vrtx::Refl_t material = intToMaterial(sphere.refl_t);
    if (!eq(color, boxes[i].col) || !eq(emission, boxes[i].emi) || !eq(minPosition, boxes[i].min) ||
        !eq(maxPosition, boxes[i].max) || material != boxes[i].refl) {
      boxes[i].col = color;
      boxes[i].emi = emission;
      boxes[i].min = minPosition;
      boxes[i].max = maxPosition;
      boxes[i].refl = material;
      boxesDirty = true;
      dirty = true;
    }
    i++;
  }
  static bool once = true;
  static int32_t oldFrame = -1;
  if (get<parameters::internal::frame>() != oldFrame) {
    oldFrame = get<parameters::internal::frame>();
    for (auto plane : get<parameters::moving_plane::plane>()) {
      auto t = plane.duration;
      auto f = plane.frequency;
      auto m = plane.magnitude;
      auto p = plane.plane_position;
      auto n = plane.plane_normal;
      auto dir = plane.plane_direction;
      auto idx = plane.index;
      p += dir * m * sinf(2.f * CUDART_PI_F * f * get<parameters::internal::simulationTime>());
      vrtx::Box mBox;
      mBox.col = float3{1, 1, 1};
      mBox.emi = float3{0, 0, 0};
      mBox.refl = vrtx::Refl_t::Lambertian;
      auto sign = n.x > 0.f || n.y > 0.f || n.z > 0.f ? true : false;
      auto x = abs(n.x) > 1e-4f;
      auto y = abs(n.y) > 1e-4f;
      auto z = abs(n.z) > 1e-4f;

      if (sign) {
        if (x) {
          mBox.max = float3{p.x, FLT_MAX, FLT_MAX};
          mBox.min = float3{p.x - 100.f, -FLT_MAX, -FLT_MAX};
        }
        if (y) {
          mBox.max = float3{FLT_MAX, p.y, FLT_MAX};
          mBox.min = float3{-FLT_MAX, p.y - 100.f, -FLT_MAX};
        }
        if (z) {
          mBox.max = float3{FLT_MAX, FLT_MAX, p.z};
          mBox.min = float3{-FLT_MAX, -FLT_MAX, p.z - 100.f};
        }
      } else {
        if (x) {
          mBox.max = float3{-p.x + 100.f, FLT_MAX, FLT_MAX};
          mBox.min = float3{-p.x, -FLT_MAX, -FLT_MAX};
        }
        if (y) {
          mBox.max = float3{FLT_MAX, -p.y + 100.f, FLT_MAX};
          mBox.min = float3{-FLT_MAX, -p.y, -FLT_MAX};
        }
        if (z) {
          mBox.max = float3{FLT_MAX, FLT_MAX, -p.z + 100.f};
          mBox.min = float3{-FLT_MAX, -FLT_MAX, -p.z};
        }
      }
      boxes[i] = mBox;
      // boxes.push_back(mBox);
      boxesDirty = true;
      dirty = true;
      // std::cout << mBox.min << " -> " << mBox.max << "" << std::endl;
      i++;
    }
    // auto t = plane.duration.value;
    // auto f = plane.frequency.value;
    // auto m = plane.magnitude.value;
    // auto p = plane.plane_position.value;
    // auto n = plane.plane_normal.value;
    // auto dir = plane.plane_direction.value;
    // auto idx = plane.index.value;
    // p += dir * m * sinf(2.f * CUDART_PI_F * f * get<parameters::internal::simulationTime>());

    // auto p_prev =
    //    plane.plane_position.value +
    //    dir * m * sinf(2.f * CUDART_PI_F * f * (get<parameters::internal::simulationTime>() - get<parameters::internal::timestep>()));
    // auto p_diff = p - p_prev;
    // auto v_diff = -p_diff / get<parameters::internal::timestep>();

    // auto nn = math::normalize(n);
    // auto d = math::dot3(p, nn);
    // float4_u<> E{nn.x, nn.y, nn.z, d};

    // if (t < get<parameters::internal::simulationTime>() && t > 0.f)
    //  continue;
    // uFloat4<SI::velocity> v{math::castTo<float4>(v_diff)};
    // launch<updatePlane>(1, 1, mem, E, v, idx);
  }
  once = false;
}
#include <tools/pathfinder.h>
vRTXrender::vRTXrender() : RTXRender() {
  cuda_particleSystem::instance().retainArray("auxHashMap");
  cuda_particleSystem::instance().retainArray("auxCellSpan");
  // cuda_particleSystem::instance().retainArray("auxCellInformation");
  // cuda_particleSystem::instance().retainArray("auxCellSurface");
  // cuda_particleSystem::instance().retainArray("auxIsoDensity");

  cuda_particleSystem::instance().retainArray("renderArray");
  cuda_particleSystem::instance().retainArray("compactCellSpan");
  cuda_particleSystem::instance().retainArray("compactHashMap");
  cuda_particleSystem::instance().retainArray("MLMResolution");
  cuda_particleSystem::instance().retainArray("centerPosition");
#ifdef ANISOTROPIC_SURFACE
  if (get<parameters::modules::anisotropicSurface>()) {
    cuda_particleSystem::instance().retainArray("anisotropicMatrices");
    cuda_particleSystem::instance().retainArray("auxIsoDensity");
  }
#else
  cuda_particleSystem::instance().retainArray("auxIsoDensity");
#endif
#ifdef BITFIELD_STRUCTURES
  cuda_particleSystem::instance().retainArray("auxLength");
#endif

  cuda_particleSystem::instance().retainArray("density");
  // std::cout << get<parameters::modules::renderMode>()
  renderMode = get<parameters::modules::renderMode>() % renderModes;
  if (renderMode == 2) {
    fluidMaterial = vrtx::Refl_t::Lambertian;
  } else
    fluidMaterial = vrtx::Refl_t::Lambertian;
  updateRTX();
  int32_t image_width, image_height;
  unsigned char* image_data = stbi_load(resolveFile("cfg/ash_uvgrid01.jpg").string().c_str(), 
      &image_width, &image_height, NULL, 4);
  float4* img = new float4[image_width * image_height];


  for (int32_t it = 0; it < image_width * image_height; ++it) {
      img[it] = float4{
      (float)image_data[it * 4 + 0],
      (float)image_data[it * 4 + 1],
      (float)image_data[it * 4 + 2], 255.f };
      //std::cout << it << ": [ " << img[it].x << " " << img[it].y << " " << img[it].z <<
      //    " ]" << std::endl;
  }


  int width = image_width;
  int height = image_height;
  int32_t size = width * height * (int32_t)sizeof(float4);
  float4 *texture = (float4 *)malloc(size);
  for (int32_t i = 0; i < width; ++i) {
    for (int32_t j = 0; j < height; ++j) {
      auto px = img[i+ (height - j - 1) * width];
      texture[i + j * width] = float4{px.x / 255.f, px.y / 255.f, px.z / 255.f, 1.f};
    }
  }

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc, width, height);

  // Copy to device memory some data located at address h_data
  // in host memory
  cudaMemcpyToArray(cuArray, 0, 0, texture, size, cudaMemcpyHostToDevice);

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture object
  texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  free(texture);
}
void vRTXrender::renderRTX(bool prettyRender, int32_t fn, int32_t s) {
  static std::vector<Matrix4x4> Txs;
  // std::cout << "vrtxRender [" << framenumber << "] @ renderMode " << renderMode << ", simulation frame: " <<
  // get<parameters::internal::frame>() << std::endl;
  static bool once = true;
  static int32_t frame = -1;
  if (once || frame != get<parameters::internal::frame>()) {
    bool update = once;
    if (once) {
      for (int32_t b = 0; b < get<parameters::boundary_volumes::volumeBoundaryCounter>(); ++b) {
        Matrix4x4 Tx;
        cudaMemcpy(&Tx, arrays::volumeBoundaryTransformMatrix::ptr + b, sizeof(Matrix4x4), cudaMemcpyDeviceToHost);
        Txs.push_back(Tx);
      }
    }
    if (frame != get<parameters::internal::frame>()) {
      for (int32_t b = 0; b < get<parameters::boundary_volumes::volumeBoundaryCounter>(); ++b) {
        Matrix4x4 Tx;
        cudaMemcpy(&Tx, arrays::volumeBoundaryTransformMatrix::ptr + b, sizeof(Matrix4x4), cudaMemcpyDeviceToHost);
        for (int32_t i = 0; i < 16; ++i)
          if (Tx.data[i] != Txs[b].data[i])
            update = true;
      }
    }
    frame = get<parameters::internal::frame>();
    if (update) {
      if (!once) {
        loader.reset();
      }
      for (int32_t b = 0; b < get<parameters::boundary_volumes::volumeBoundaryCounter>(); ++b) {
        loader.appendObject(b);
      }
      objects = loader.mergeMeshes();
      loader.tearDownMeshes();
      loader.buildBVH();
    }
    once = false;
  }
  float3 anisoH = float3{get<parameters::render_settings::maxAnisotropicSupport>().x, get<parameters::render_settings::maxAnisotropicSupport>().y,
                         get<parameters::render_settings::maxAnisotropicSupport>().z};
  auto cs = anisoH;
  // get<parameters::modules::anisotropicSurface>() ? anisoH : get<parameters::internal::cell_size>();
  vrtxFluidArrays farrays;
  farrays.min_coord = get<parameters::internal::min_coord>();
  farrays.max_coord = get<parameters::internal::max_coord>() - anisoH + cs;
  farrays.cell_size = cs;
  // farrays.gridSize = get<parameters::internal::gridSize>();
  farrays.hash_entries = get<parameters::simulation_settings::hash_entries>();
  farrays.mlm_schemes = get<parameters::simulation_settings::mlm_schemes>();
  farrays.num_ptcls = get<parameters::internal::num_ptcls>();
  farrays.maxNumptcls = get<parameters::simulation_settings::maxNumptcls>();
  farrays.timestep = get<parameters::internal::timestep>();
  farrays.renderRadius =
      get<parameters::particle_settings::radius>() * get<parameters::render_settings::vrtxWMin>(); // *get<parameters::render_settings::vrtxWMin>(); //(get<parameters::modules::adaptive>()
                            //? 1.f/powf((float)get<parameters::adaptive::resolution>(), 1.f/3.f) : 1.f);
  // farrays.renderRadius = get<parameters::particle_settings::radius>();
  farrays.rest_density = get<parameters::particle_settings::rest_density>();

  farrays.compactHashMap = arrays::compactHashMap::ptr;
  farrays.compactCellSpan = arrays::compactCellSpan::ptr;
  farrays.auxLength = arrays::auxLength::ptr;
  farrays.MLMResolution = arrays::MLMResolution::ptr;
  farrays.position = arrays::centerPosition::ptr;
  farrays.volume = arrays::volume::ptr;
  farrays.density = arrays::density::ptr;
  farrays.renderArray = arrays::renderArray::ptr;
  farrays.auxIsoDensity = arrays::auxIsoDensity::ptr;

  farrays.centerPosition = arrays::centerPosition::ptr;
  farrays.anisotropicMatrices = arrays::anisotropicMatrices::ptr;

  farrays.minMap = get<parameters::color_map::min>();
  farrays.maxMap = get<parameters::color_map::max>();
  farrays.transferFn = get<parameters::color_map::transfer_fn>();

  vrtxFluidMemory fmem;
  // fmem.gridSize = get<parameters::internal::gridSize>();
  fmem.cell_size = cs;
  fmem.min_coord = get<parameters::internal::min_coord>();
  fmem.max_coord = get<parameters::internal::max_coord>();// +cs * get<parameters::render_settings::auxScale>();

  fmem.hash_entries = get<parameters::simulation_settings::hash_entries>();
  fmem.mlm_schemes = get<parameters::simulation_settings::mlm_schemes>();
  fmem.uvmap = texObj;
  fmem.num_ptcls = get<parameters::internal::num_ptcls>();
  fmem.max_numptcls = get<parameters::simulation_settings::maxNumptcls>();
  fmem.timestep = get<parameters::internal::timestep>();
  fmem.renderRadius = get<parameters::particle_settings::radius>() * get<parameters::render_settings::vrtxWMin>();// (get<parameters::modules::adaptive>() ? 1.f /
                                            // powf((float)get<parameters::adaptive::resolution>(), 1.f / 3.f) : 1.f);
  // fmem.renderRadius = get<parameters::particle_settings::radius>();
  fmem.rest_density = get<parameters::particle_settings::rest_density>();
  fmem.fluidBias = get<parameters::render_settings::vrtxFluidBias>();

  fmem.cellSpan = arrays::auxCellSpan::ptr;
  fmem.hashMap = arrays::auxHashMap::ptr;

  fmem.vrtxR = get<parameters::render_settings::vrtxR>();
  fmem.bounces = get<parameters::render_settings::vrtxBounces>();

  fmem.wmin = get<parameters::render_settings::anisotropicKs>();
  fmem.wmax = get<parameters::render_settings::vrtxWMax>();

  fmem.bvhColor = get<parameters::render_settings::bvhColor>();
  fmem.vrtxDepth = get<parameters::render_settings::vrtxDepth>();
  fmem.vrtxNormal = get<parameters::render_settings::vrtxRenderNormals>();
  fmem.vrtxSurface = get<parameters::render_settings::vrtxRenderSurface>();
  fmem.vrtxDepthScale = get<parameters::render_settings::vrtxDepthScale>();
  fmem.colorMapFlipped = get<parameters::color_map::map_flipped>();
  fmem.colorMap = cu_color_map;
  fmem.colorMapLength = color_map_elements - 1;
  fmem.IOR = get<parameters::render_settings::vrtxIOR>();
  fmem.renderFloor = get<parameters::render_settings::floorRender>() == 1;
  fmem.auxScale = get<parameters::render_settings::auxScale>();
  // fmem.cell_size *= fmem.auxScale;
  // fmem.gridSize /= static_cast<int32_t>(fmem.auxScale);
  // fmem.gridSize += 1;
  // fmem.cell_size_actual = get<parameters::internal::cell_size>();
  // fmem.gridSize_actual = get<parameters::internal::gridSize>();

  fmem.maxZ_coordx = (position_to_morton_32(fmem.max_coord, fmem) & 0b001001001001001001001001001001u);
  fmem.maxZ_coordy = (position_to_morton_32(fmem.max_coord, fmem) & 0b010010010010010010010010010010u);
  fmem.maxZ_coordz = (position_to_morton_32(fmem.max_coord, fmem) & 0b100100100100100100100100100100u);

  fmem.vrtxFluidColor = get<parameters::render_settings::vrtxFluidColor>();
  fmem.vrtxDebeer = get<parameters::render_settings::vrtxDebeer>();
  fmem.vrtxDebeerScale = get<parameters::render_settings::vrtxDebeerScale>();
  fmem.bvhMaterial = intToMaterial(get<parameters::render_settings::vrtxBVHMaterial>());
  fmem.fluidMaterial = intToMaterial(get<parameters::render_settings::vrtxMaterial>());
  fmem.surfaceTechnique = get<parameters::render_settings::vrtxSurfaceExtraction>();

  fmem.vrtxDomainMin = get<parameters::render_settings::vrtxDomainMin>();
  fmem.vrtxDomainMax = get<parameters::render_settings::vrtxDomainMax>();
  fmem.vrtxDomainEpsilon = get<parameters::render_settings::vrtxDomainEpsilon>();
  fmem.vrtxRenderDomainMax = get<parameters::render_settings::vrtxRenderDomainMax>();
  fmem.vrtxRenderDomainMin = get<parameters::render_settings::vrtxRenderDomainMin>();

  // std::cout << fmem.maxZ_coordx << std::endl;
  // std::cout << fmem.maxZ_coordy << std::endl;
  // std::cout << fmem.maxZ_coordz << std::endl;

  timings =
      cuVRTXRender(hostScene(), renderedResourceOut, loader, fmem, farrays, accumulatebuffer, fn, s, renderMode,
                   bounces, fluidRender, get<parameters::render_settings::vrtxRenderGrid>(), get<parameters::render_settings::vrtxSurfaceExtraction>(),
                   intToMaterial(get<parameters::render_settings::vrtxMaterial>()), boxesDirty, spheres, boxes);
  boxesDirty = false;
}