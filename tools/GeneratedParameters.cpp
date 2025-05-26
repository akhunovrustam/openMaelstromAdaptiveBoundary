#include <math/math.h>
#include <tools/ParameterManager.h>
#include <utility/identifier/uniform.h>
#include <utility/identifier/arrays.h>
#include <tools/pathfinder.h>
#pragma warning(push)
#pragma warning(disable : 4251; disable : 4275)
#include <yaml-cpp/yaml.h>
#pragma warning(pop)
#include <imgui/imgui.h>

void ParameterManager::init() {
ParameterManager::instance().newParameter("adaptive.adaptivityScaling", float{0.825f},{
		.description = "",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("adaptive.adaptivityThreshold", float{1.0f},{
		.description = "",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("adaptive.adaptivityGamma", float{0.1f},{
		.description = "",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
static auto adaptive_resolution_min = 1.f;
static auto adaptive_resolution_max = 512.f;
ParameterManager::instance().newParameter("adaptive.resolution", float{32.f},{
		.description = "Target adaptivity ratio of volumes for the simulation. Useful value: 16.f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{adaptive_resolution_min, adaptive_resolution_max},
	});
ParameterManager::instance().newParameter("adaptive.useVolume", int32_t{1},{
		.description = ".",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("adaptive.minVolume", float{1.f},{
		.description = ".",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::volume>)},
	});
static auto adaptive_detailedAdaptiveStatistics_min = 0;
static auto adaptive_detailedAdaptiveStatistics_max = 1;
ParameterManager::instance().newParameter("adaptive.detailedAdaptiveStatistics", int32_t{1},{
		.description = ".",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{adaptive_detailedAdaptiveStatistics_min, adaptive_detailedAdaptiveStatistics_max},
	});
ParameterManager::instance().newParameter("adaptive.ratio", float{1.f},{
		.description = "Target adaptivity ratio of volumes for the simulation. Useful value: 16.f.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("adaptive.blendSteps", float{10.f},{
		.description = "Timesteps that the simulation should blend split particles over. Useful value: 10.f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("adaptive.delay", float{1.f},{
		.description = "Initial delay the simulation should wait before adjusting the resolution due to initial errors. Useful value: 1.f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::s>)},
	});
ParameterManager::instance().newParameter("adaptive.splitPtcls", std::vector<int32_t>{0},{
		.description = "Value that represents the number of particles split in the last splitting step.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("adaptive.blendedPtcls", int32_t{0},{
		.description = "Value that represents the number of particles split in the last splitting step.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("adaptive.mergedPtcls", std::vector<int32_t>{0},{
		.description = "Value that represents the number of particles merged in the last merge step.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("adaptive.sharedPtcls", std::vector<int32_t>{0},{
		.description = "Value that represents the number of particles sharing in the last share step.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("alembic.file_name", std::string{"export/alembic_$f.abc"},{
		.description = "File name scheme to export the particle data to. $f in the name marks a wildcard for the frame number, e.g. export/alembic_$f.abc",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("alembic.fps", int32_t{24},{
		.description = "Framerate at which particles should be exported to disk. Useful value: 24.f.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("boundary_volumes.volumeBoundaryCounter", int32_t{0},{
		.description = "",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("boundary_volumes.volume", std::vector<boundaryVolume>{},{
		.description = "Complex parameter that describes a volume that should be loaded from a file and used as a one way coupled boundary Object.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
static std::vector<std::string> color_map_transfer_mode_presets = {std::string("linear"),std::string("cubicRoot"),std::string("cubic"),std::string("squareRoot"),std::string("square"),std::string("log")};
ParameterManager::instance().newParameter("color_map.transfer_mode", std::string{"linear"},{
		.description = "Function applied to value before applying colormap",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.presets = color_map_transfer_mode_presets
	});
static std::vector<std::string> color_map_mapping_mode_presets = {std::string("linear"),std::string("cubicRoot"),std::string("cubic"),std::string("squareRoot"),std::string("square"),std::string("log")};
ParameterManager::instance().newParameter("color_map.mapping_mode", std::string{"linear"},{
		.description = "Function applied to value before applying colormap",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.presets = color_map_mapping_mode_presets
	});
static std::vector<std::string> color_map_vectorMode_presets = {std::string("length"),std::string("x"),std::string("y"),std::string("z"),std::string("w")};
ParameterManager::instance().newParameter("color_map.vectorMode", std::string{"length"},{
		.description = "Function applied to value before applying colormap",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.presets = color_map_vectorMode_presets
	});
static auto color_map_visualizeDirection_min = 0;
static auto color_map_visualizeDirection_max = 1;
ParameterManager::instance().newParameter("color_map.visualizeDirection", int32_t{0},{
		.description = "Function applied to value before applying colormap",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{color_map_visualizeDirection_min, color_map_visualizeDirection_max},
	});
static auto color_map_vectorScale_min = 0.f;
static auto color_map_vectorScale_max = 10.f;
ParameterManager::instance().newParameter("color_map.vectorScale", float{1},{
		.description = "Function applied to value before applying colormap",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{color_map_vectorScale_min, color_map_vectorScale_max},
	});
static auto color_map_vectorScaling_min = 0;
static auto color_map_vectorScaling_max = 1;
ParameterManager::instance().newParameter("color_map.vectorScaling", int32_t{0},{
		.description = "Represents the lower boundary of the color mapping. Useful values depend on what is being visualized.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{color_map_vectorScaling_min, color_map_vectorScaling_max},
	});
static auto color_map_min_min = -10.f;
static auto color_map_min_max = 10.f;
ParameterManager::instance().newParameter("color_map.min", float{0.f},{
		.description = "Represents the lower boundary of the color mapping. Useful values depend on what is being visualized.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{color_map_min_min, color_map_min_max},
	});
static auto color_map_max_min = -10.f;
static auto color_map_max_max = 10.f;
ParameterManager::instance().newParameter("color_map.max", float{1.f},{
		.description = "Represents the upper boundary of the color mapping. Useful values depend on what is being visualized.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{color_map_max_min, color_map_max_max},
	});
ParameterManager::instance().newParameter("color_map.transfer_fn", int32_t{0},{
		.description = "Used to enable/disable automatically scaling the visualization range to the full input range.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
static auto color_map_pruneVoxel_min = 0;
static auto color_map_pruneVoxel_max = 1;
ParameterManager::instance().newParameter("color_map.pruneVoxel", int32_t{0},{
		.description = "Used to enable/disable automatically scaling the visualization range to the full input range.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{color_map_pruneVoxel_min, color_map_pruneVoxel_max},
	});
ParameterManager::instance().newParameter("color_map.mapping_fn", int32_t{0},{
		.description = "Used to enable/disable automatically scaling the visualization range to the full input range.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
static auto color_map_autoScaling_min = 0;
static auto color_map_autoScaling_max = 1;
ParameterManager::instance().newParameter("color_map.autoScaling", int{1},{
		.description = "Used to enable/disable automatically scaling the visualization range to the full input range.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{color_map_autoScaling_min, color_map_autoScaling_max},
	});
static auto color_map_map_flipped_min = 0;
static auto color_map_map_flipped_max = 1;
ParameterManager::instance().newParameter("color_map.map_flipped", int{0},{
		.description = "Used to enable/disable automatically scaling the visualization range to the full input range.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{color_map_map_flipped_min, color_map_map_flipped_max},
	});
ParameterManager::instance().newParameter("color_map.buffer", std::string{"density"},{
		.description = "Contains the name of the array that should be visualized, e.g. density.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
static std::vector<std::string> color_map_map_presets = []() {std::vector <std::string> colorMaps;auto f = std::filesystem::path(ParameterManager::instance().get<std::string>("stylePath"));auto p = f.parent_path().string();if (*(p.end() - 1) == '/' || *(p.end() - 1) == '\\')p = p.substr(0, p.length() - 1);std::replace(p.begin(), p.end(), '\\', '/');for (auto& p : std::filesystem::directory_iterator(p))if (p.path().extension().string().find(".png") != std::string::npos)colorMaps.push_back(p.path().filename().replace_extension("").string());return colorMaps; }();
ParameterManager::instance().newParameter("color_map.map", std::string{"inferno"},{
		.description = "Path to a file that contains the color map used for visualization.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.presets = color_map_map_presets
	});
ParameterManager::instance().newParameter("dfsph_settings.densityError", float{0.f},{
		.description = "Contains the average density error of the last solve step in percent.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("dfsph_settings.divergenceError", float{0.f},{
		.description = "Contains the average density error of the last solve step in percent.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::Hz>)},
	});
ParameterManager::instance().newParameter("dfsph_settings.densitySolverIterations", int32_t{0},{
		.description = "Contains the number of iterations for the DFSPH solver.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("dfsph_settings.divergenceSolverIterations", int32_t{0},{
		.description = "Contains the number of iterations for the DFSPH solver.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("dfsph_settings.densityEta", float{0.0001f},{
		.description = "Maximum average density error allowed for the incompressibility solver in percent. Useful value: 0.01f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("dfsph_settings.divergenceEta", float{0.001f},{
		.description = "Maximum average density error allowed for the divergence free solver in percent. Useful value: 0.01f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::Hz>)},
	});
ParameterManager::instance().newParameter("iisph_settings.density_error", float{0.f},{
		.description = "Contains the average density error of the last solve step in percent.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("iisph_settings.iterations", int32_t{0},{
		.description = "Contains the number of iterations for the IISPH solver.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("iisph_settings.eta", float{0.1f},{
		.description = "Maximum average density error allowed for the incompressibility solver in percent. Useful value: 0.01f.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("iisph_settings.jacobi_omega", float{0.2f},{
		.description = "Relaxation factor for the relaxed jacobi solvler. Useful value: 0.5f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("inlet_volumes.volume", std::vector<inletVolume>{},{
		.description = "Complex parameter that describes a volume that should be loaded from a file and emitted as particles continuously.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.neighborhood_kind", neighbor_list{neighbor_list::constrained},{
		.description = "Internal value that represents the used neighborlist, not visible in the simulation as it is an enum.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.dumpNextframe", int32_t{0},{
		.description = "Internal value that represents the used launch configuration, not visible in the simulation as it is an enum.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.dumpForSSSPH", int32_t{0},{
		.description = "Internal value that represents the used launch configuration, not visible in the simulation as it is an enum.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.target", launch_config{launch_config::device},{
		.description = "Internal value that represents the used launch configuration, not visible in the simulation as it is an enum.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.hash_size", hash_length{hash_length::bit_64},{
		.description = "Internal value that represents the used morton code length, not visible in the simulation as it is an enum.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.cell_ordering", cell_ordering{cell_ordering::z_order},{
		.description = "Internal value that represents the used cell ordering, not visible in the simulation as it is an enum.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.cell_structure", cell_structuring{cell_structuring::hashed},{
		.description = "Internal value that represents the used cell structure, not visible in the simulation as it is an enum.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.num_ptcls", int32_t{0u},{
		.description = "Represents the current number of particles valid within the simulation.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.num_ptcls_fluid", int32_t{0u},{
		.description = "Represents the current number of fluid particles valid within the simulation.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.folderName", std::string{"logDump"},{
		.description = "Represents the current number of fluid particles valid within the simulation.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.boundaryCounter", int32_t{0u},{
		.description = "Represents the number of active boundary planes.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.boundaryLUTSize", int32_t{0u},{
		.description = "Represents the size of the boundary LUT.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.frame", int32_t{0u},{
		.description = "Represents the current frame where frame is the number of timesteps not related to any export fps.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.max_velocity", float{1.f},{
		.description = "Contains the length of the fastest particles velocity.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::velocity>)},
	});
ParameterManager::instance().newParameter("internal.min_domain", float3{0.f,0.f,0.f},{
		.description = "Lower bound of the simulation domain. Calculated based on the boundary object and fluid domain",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat3<SI::m>)},
	});
ParameterManager::instance().newParameter("internal.max_domain", float3{0.f,0.f,0.f},{
		.description = "Upper bound of the simulation domain. Calculated based on the boundary object and fluid domain",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat3<SI::m>)},
	});
ParameterManager::instance().newParameter("internal.min_coord", float3{0.f,0.f,0.f},{
		.description = "Lower bound of the particle coordinates. Based purely on the fluid domain.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat3<SI::m>)},
	});
ParameterManager::instance().newParameter("internal.max_coord", float3{0.f,0.f,0.f},{
		.description = "Upper bound of the particle coordinates. Based purely on the fluid domain.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat3<SI::m>)},
	});
ParameterManager::instance().newParameter("internal.cell_size", float3{0.f,0.f,0.f},{
		.description = "Represents the size of a cell which depends on the lowest particle resolutions default support radius.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat3<SI::m>)},
	});
ParameterManager::instance().newParameter("internal.gridSize", int3{0u,0u,0u},{
		.description = "Represents the size of the active simulation domain in voxels. Not representative of the memory consumption with hashed methods.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.ptcl_spacing", float{0.f},{
		.description = "Value used when creating particles on a hexagonal grid as a multiplier to the grid coordinates to achieve the most dense packing.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::m>)},
	});
ParameterManager::instance().newParameter("internal.ptcl_support", float{0.f},{
		.description = "Default support radius of the lowest resolution of particles.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::m>)},
	});
ParameterManager::instance().newParameter("internal.config_file", std::string{"DamBreakObstacle.json"},{
		.description = "The file that was loaded to initialize the simulation.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.config_folder", std::string{"D:/DamBreak"},{
		.description = "The path to the configuration file being used for the simulation.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.working_directory", std::string{"D:/DamBreak"},{
		.description = "The path to the configuration file being used for the simulation.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.build_directory", std::string{"D:/DamBreak"},{
		.description = "The path to the configuration file being used for the simulation.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.source_directory", std::string{"D:/DamBreak"},{
		.description = "The path to the configuration file being used for the simulation.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.binary_directory", std::string{"D:/DamBreak"},{
		.description = "The path to the configuration file being used for the simulation.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("internal.timestep", float{0.f},{
		.description = "The current timestep of the simulation defined by the CFL condition and the min/max timestep values.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::s>)},
	});
ParameterManager::instance().newParameter("internal.simulationTime", float{0.f},{
		.description = "Represents the simulated time",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::s>)},
	});
ParameterManager::instance().newParameter("modules.adaptive", bool{true},{
		.description = "Used to enable/disable continuous adaptivity, requires surfaceDistance = true.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.pressure", std::string{"DFSPH"},{
		.description = "Used to select the incompressibility solver. IISPH, IISPH17 and DFSPH are valid options.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.volumeBoundary", bool{true},{
		.description = "Used to enable/disable openVDB based rigid objects.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.xsph", bool{true},{
		.description = "Used to enable/disable XSPH viscosity forces.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.drag", std::string{"Gissler17"},{
		.description = "Used to enable/disable air drag forces.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.viscosity", bool{false},{
		.description = "Used to enable/disable artificial viscosity, helps with adaptivity.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.tension", std::string{"Akinci"},{
		.description = "Used to select the surface tension algorithm. Akinci and none are valid options.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.vorticity", std::string{"Bender17"},{
		.description = "Used to select the Vorticity algorithm. Bender17 and none are valid options.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.movingBoundaries", bool{false},{
		.description = "Used to enable/disable correct error checking after each kernel call which introduces some overhead.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.debug", bool{false},{
		.description = "Used to enable/disable continuous adaptivity, requires surfaceDistance = true.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.density", std::string{"standard"},{
		.description = "Used to select the underlying resorting algorithm. linear_cell and hashed_cell are valid options.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.particleCleanUp", bool{true},{
		.description = "Used to enable/disable correct error checking after each kernel call which introduces some overhead.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.volumeInlets", bool{false},{
		.description = "Used to enable/disable correct error checking after each kernel call which introduces some overhead.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.volumeOutlets", bool{false},{
		.description = "Used to enable/disable correct error checking after each kernel call which introduces some overhead.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.logDump", std::string{"none"},{
		.description = "Used to enable/disable log dump of physical quantities.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.neighborhood", std::string{"constrained"},{
		.description = "Used to select the neighborhood search algorithm. constrained, basic and cell_based are valid options.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.neighborSorting", bool{false},{
		.description = "Used to select the neighborhood search algorithm. constrained, basic and cell_based are valid options.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.rayTracing", bool{true},{
		.description = "Used to enable/disable rayTracing using auxMLM.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.anisotropicSurface", bool{true},{
		.description = "Used to enable/disable rayTracing using auxMLM.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.renderMode", int32_t{0},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.resorting", std::string{"hashed_cell"},{
		.description = "Used to select the underlying resorting algorithm. linear_cell and hashed_cell are valid options.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.hash_width", std::string{"64bit"},{
		.description = "Used to select the length of Morton codes being used. 32bit and 64bit are valid options, MLM requires 64bit. ",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.alembic", bool{false},{
		.description = "Used to enable/disable exporting particles to alembic files.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.error_checking", bool{false},{
		.description = "Used to enable/disable correct error checking after each kernel call which introduces some overhead.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.gl_record", bool{false},{
		.description = "Used to enable/disable recording of the gl viewport into a file.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.launch_cfg", std::string{"device"},{
		.description = "Used to select where the code should be run. Valid options are: gpu (runs almost everything on the gpu), cpu (runs almost everything on the cpu) and debug (same as cpu but single threaded).",
		.constant = true,
		.hidden = true,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.regex_cfg", bool{false},{
		.description = "",
		.constant = false,
		.hidden = true,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.support", std::string{"constrained"},{
		.description = "Used to select the algorithm used to constrain particle support. constrained and none are valid options.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.surfaceDistance", bool{false},{
		.description = "Used to enable/disable surface distance calculations, required for adaptivity.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("modules.surfaceDetection", bool{true},{
		.description = "Used to enable/disable surface distance calculations, required for adaptivity.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("moving_plane.plane", std::vector<movingPlane>{},{
		.description = "Complex parameter that describes a moving boundary Wall.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("outlet_volumes.volumeOutletCounter", int32_t{0},{
		.description = "Represents the number of valid boundary volumes.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("outlet_volumes.volumeOutletTime", float{-1.f},{
		.description = "Represents the number of valid boundary volumes.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::s>)},
	});
ParameterManager::instance().newParameter("outlet_volumes.volume", std::vector<outletVolume>{},{
		.description = "Complex parameter that describes a volume that should be loaded from a file and emitted as particles continuously.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("particleSets.set", std::vector<std::string>{},{
		.description = "Complex parameter that describes a volume that should be loaded from a file and emitted as particles continuously.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
static auto particle_settings_monaghan_viscosity_min = 0.000f;
static auto particle_settings_monaghan_viscosity_max = 50.f;
ParameterManager::instance().newParameter("particle_settings.monaghan_viscosity", float{5.f},{
		.description = "Artificial viscosity strength parameter. Useful value: 15.f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::velocity>)},
		.range = Range{particle_settings_monaghan_viscosity_min, particle_settings_monaghan_viscosity_max},
	});
static auto particle_settings_boundaryViscosity_min = 0.000f;
static auto particle_settings_boundaryViscosity_max = 50.f;
ParameterManager::instance().newParameter("particle_settings.boundaryViscosity", float{0.0375f},{
		.description = "Artificial viscosity strength parameter. Useful value: 15.f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::velocity>)},
		.range = Range{particle_settings_boundaryViscosity_min, particle_settings_boundaryViscosity_max},
	});
static auto particle_settings_xsph_viscosity_min = 0.000f;
static auto particle_settings_xsph_viscosity_max = 1.f;
ParameterManager::instance().newParameter("particle_settings.xsph_viscosity", float{0.05f},{
		.description = "XSPH viscosity strenght parameter. Useful value: 0.05f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{particle_settings_xsph_viscosity_min, particle_settings_xsph_viscosity_max},
	});
static auto particle_settings_rigidAdhesion_akinci_min = 0.000f;
static auto particle_settings_rigidAdhesion_akinci_max = 2.f;
ParameterManager::instance().newParameter("particle_settings.rigidAdhesion_akinci", float{0.f},{
		.description = "Strength of the surface tension for the akinci based surface tension model. Useful value: 0.1f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::acceleration>)},
		.range = Range{particle_settings_rigidAdhesion_akinci_min, particle_settings_rigidAdhesion_akinci_max},
	});
static auto particle_settings_boundaryAdhesion_akinci_min = 0.000f;
static auto particle_settings_boundaryAdhesion_akinci_max = 2.f;
ParameterManager::instance().newParameter("particle_settings.boundaryAdhesion_akinci", float{0.f},{
		.description = "Strength of the surface tension for the akinci based surface tension model. Useful value: 0.1f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::acceleration>)},
		.range = Range{particle_settings_boundaryAdhesion_akinci_min, particle_settings_boundaryAdhesion_akinci_max},
	});
static auto particle_settings_tension_akinci_min = 0.000f;
static auto particle_settings_tension_akinci_max = 2.f;
ParameterManager::instance().newParameter("particle_settings.tension_akinci", float{0.15f},{
		.description = "Strength of the surface tension for the akinci based surface tension model. Useful value: 0.1f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::acceleration>)},
		.range = Range{particle_settings_tension_akinci_min, particle_settings_tension_akinci_max},
	});
static auto particle_settings_air_velocity_min = -10.f;
static auto particle_settings_air_velocity_max = 10.f;
ParameterManager::instance().newParameter("particle_settings.air_velocity", float4{0.f,0.f,0.f,0.f},{
		.description = "Global air velocity of the simulation for air drag effects. Useful value: {0,0,0,0}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat4<SI::velocity>)},
		.range = Range{particle_settings_air_velocity_min, particle_settings_air_velocity_max},
	});
ParameterManager::instance().newParameter("particle_settings.radius", float{0.5f},{
		.description = "Radius of the lowest resolution particles (in meters). Useful value: 0.5f.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::m>)},
	});
ParameterManager::instance().newParameter("particle_settings.first_fluid", int{-1},{
		.description = "Index of the first fluid particle",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("particle_settings.max_vel", float{-1.0f},{
		.description = "Max fluid velocity",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("particle_settings.min_vel", float{-1.0f},{
		.description = "Min fluid velocity",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("particle_settings.max_neighbors", int{-1},{
		.description = "Max number of neighbors",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("particle_settings.max_density", float{-1},{
		.description = "Max density of the boundary particles",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("particle_settings.sdf_resolution", int{20},{
		.description = "Max density of the boundary particles",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("particle_settings.sdf_epsilon", float{0.1},{
		.description = "Max density of the boundary particles",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("particle_settings.sdf_minpoint", float4{0.f,0.f,0.f,0.f},{
		.description = "min point of sdf",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("particle_settings.rest_density", float{998.f},{
		.description = "Rest density of all fluid particles. Useful value: 998.f.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::density>)},
	});
ParameterManager::instance().newParameter("particle_volumes.volume", std::vector<particleVolume>{},{
		.description = "Complex parameter that describes a volume that should be loaded from a file and emitted as particles once.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("render_settings.maxAnisotropicSupport", float4{1.f, 1.f, 1.f, 1.f},{
		.description = ".",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
static auto render_settings_apertureRadius_min = 0.f;
static auto render_settings_apertureRadius_max = 2.f;
ParameterManager::instance().newParameter("render_settings.apertureRadius", float{0.15f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_apertureRadius_min, render_settings_apertureRadius_max},
	});
static auto render_settings_anisotropicLambda_min = 0.f;
static auto render_settings_anisotropicLambda_max = 1.f;
ParameterManager::instance().newParameter("render_settings.anisotropicLambda", float{0.980198f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_anisotropicLambda_min, render_settings_anisotropicLambda_max},
	});
static auto render_settings_anisotropicNepsilon_min = 0;
static auto render_settings_anisotropicNepsilon_max = 60;
ParameterManager::instance().newParameter("render_settings.anisotropicNepsilon", int32_t{40},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_anisotropicNepsilon_min, render_settings_anisotropicNepsilon_max},
	});
static auto render_settings_anisotropicKs_min = 0.f;
static auto render_settings_anisotropicKs_max = 2.f;
ParameterManager::instance().newParameter("render_settings.anisotropicKs", float{1.f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_anisotropicKs_min, render_settings_anisotropicKs_max},
	});
static auto render_settings_anisotropicKr_min = 0.f;
static auto render_settings_anisotropicKr_max = 10.f;
ParameterManager::instance().newParameter("render_settings.anisotropicKr", float{4.0f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_anisotropicKr_min, render_settings_anisotropicKr_max},
	});
static auto render_settings_anisotropicKn_min = 0.f;
static auto render_settings_anisotropicKn_max = 2.f;
ParameterManager::instance().newParameter("render_settings.anisotropicKn", float{0.188806f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_anisotropicKn_min, render_settings_anisotropicKn_max},
	});
static auto render_settings_focalDistance_min = 0.f;
static auto render_settings_focalDistance_max = 100.f;
ParameterManager::instance().newParameter("render_settings.focalDistance", float{100.f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_focalDistance_min, render_settings_focalDistance_max},
	});
static auto render_settings_vrtxNeighborLimit_min = 0;
static auto render_settings_vrtxNeighborLimit_max = 100;
ParameterManager::instance().newParameter("render_settings.vrtxNeighborLimit", int32_t{0},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxNeighborLimit_min, render_settings_vrtxNeighborLimit_max},
	});
static auto render_settings_vrtxFluidBias_min = 0.f;
static auto render_settings_vrtxFluidBias_max = 2.5f;
ParameterManager::instance().newParameter("render_settings.vrtxFluidBias", float{0.05f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxFluidBias_min, render_settings_vrtxFluidBias_max},
	});
static auto render_settings_vrtxRenderDomainMin_min = -200.f;
static auto render_settings_vrtxRenderDomainMin_max = 200.f;
ParameterManager::instance().newParameter("render_settings.vrtxRenderDomainMin", float3{-100.f, -100.f, 0.f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxRenderDomainMin_min, render_settings_vrtxRenderDomainMin_max},
	});
static auto render_settings_vrtxRenderDomainMax_min = -200.f;
static auto render_settings_vrtxRenderDomainMax_max = 200.f;
ParameterManager::instance().newParameter("render_settings.vrtxRenderDomainMax", float3{100.f, 100.f, 200.f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxRenderDomainMax_min, render_settings_vrtxRenderDomainMax_max},
	});
ParameterManager::instance().newParameter("render_settings.vrtxFlipCameraUp", int32_t{0},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("render_settings.vrtxSurfaceExtraction", int32_t{0},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("render_settings.vrtxRenderMode", int32_t{0},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
static auto render_settings_vrtxRenderGrid_min = 0;
static auto render_settings_vrtxRenderGrid_max = 1;
ParameterManager::instance().newParameter("render_settings.vrtxRenderGrid", int32_t{0},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxRenderGrid_min, render_settings_vrtxRenderGrid_max},
	});
static auto render_settings_vrtxRenderFluid_min = 0;
static auto render_settings_vrtxRenderFluid_max = 1;
ParameterManager::instance().newParameter("render_settings.vrtxRenderFluid", int32_t{1},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxRenderFluid_min, render_settings_vrtxRenderFluid_max},
	});
static auto render_settings_vrtxRenderSurface_min = 0;
static auto render_settings_vrtxRenderSurface_max = 1;
ParameterManager::instance().newParameter("render_settings.vrtxRenderSurface", int32_t{1},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxRenderSurface_min, render_settings_vrtxRenderSurface_max},
	});
static auto render_settings_vrtxDisplayStats_min = 0;
static auto render_settings_vrtxDisplayStats_max = 1;
ParameterManager::instance().newParameter("render_settings.vrtxDisplayStats", int32_t{1},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxDisplayStats_min, render_settings_vrtxDisplayStats_max},
	});
static auto render_settings_vrtxRenderBVH_min = 0;
static auto render_settings_vrtxRenderBVH_max = 1;
ParameterManager::instance().newParameter("render_settings.vrtxRenderBVH", int32_t{1},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxRenderBVH_min, render_settings_vrtxRenderBVH_max},
	});
static auto render_settings_vrtxBVHMaterial_min = 0;
static auto render_settings_vrtxBVHMaterial_max = 4;
ParameterManager::instance().newParameter("render_settings.vrtxBVHMaterial", int32_t{1},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxBVHMaterial_min, render_settings_vrtxBVHMaterial_max},
	});
static auto render_settings_vrtxRenderNormals_min = 0;
static auto render_settings_vrtxRenderNormals_max = 1;
ParameterManager::instance().newParameter("render_settings.vrtxRenderNormals", int32_t{0},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxRenderNormals_min, render_settings_vrtxRenderNormals_max},
	});
static auto render_settings_vrtxMaterial_min = 0;
static auto render_settings_vrtxMaterial_max = 4;
ParameterManager::instance().newParameter("render_settings.vrtxMaterial", int32_t{0},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxMaterial_min, render_settings_vrtxMaterial_max},
	});
static auto render_settings_vrtxDomainEpsilon_min = -3.f;
static auto render_settings_vrtxDomainEpsilon_max = 3.f;
ParameterManager::instance().newParameter("render_settings.vrtxDomainEpsilon", float{-1.762063f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxDomainEpsilon_min, render_settings_vrtxDomainEpsilon_max},
	});
ParameterManager::instance().newParameter("render_settings.vrtxDomainMin", float3{-1.f, -1.f, -1.f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("render_settings.vrtxDomainMax", float3{1.f, 1.f, 1.f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
static auto render_settings_vrtxDebeerScale_min = 0.001f;
static auto render_settings_vrtxDebeerScale_max = 1.f;
ParameterManager::instance().newParameter("render_settings.vrtxDebeerScale", float{0.056f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxDebeerScale_min, render_settings_vrtxDebeerScale_max},
	});
static auto render_settings_vrtxDebeer_min = 0.f;
static auto render_settings_vrtxDebeer_max = 1.f;
ParameterManager::instance().newParameter("render_settings.vrtxDebeer", float3{0.94902f, 0.76863f, 0.505823f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxDebeer_min, render_settings_vrtxDebeer_max},
	});
static auto render_settings_bvhColor_min = 0.f;
static auto render_settings_bvhColor_max = 1.f;
ParameterManager::instance().newParameter("render_settings.bvhColor", float3{0.566f, 0.621f, 0.641f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_bvhColor_min, render_settings_bvhColor_max},
	});
static auto render_settings_vrtxFluidColor_min = 0.f;
static auto render_settings_vrtxFluidColor_max = 1.f;
ParameterManager::instance().newParameter("render_settings.vrtxFluidColor", float3{0.897f, 0.917f, 1.f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxFluidColor_min, render_settings_vrtxFluidColor_max},
	});
static auto render_settings_vrtxDepth_min = 0;
static auto render_settings_vrtxDepth_max = 1;
ParameterManager::instance().newParameter("render_settings.vrtxDepth", int32_t{0},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxDepth_min, render_settings_vrtxDepth_max},
	});
static auto render_settings_vrtxDepthScale_min = 0.f;
static auto render_settings_vrtxDepthScale_max = 2.f;
ParameterManager::instance().newParameter("render_settings.vrtxDepthScale", float{0.1f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxDepthScale_min, render_settings_vrtxDepthScale_max},
	});
static auto render_settings_vrtxWMin_min = 0.f;
static auto render_settings_vrtxWMin_max = 1.f;
ParameterManager::instance().newParameter("render_settings.vrtxWMin", float{0.4f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxWMin_min, render_settings_vrtxWMin_max},
	});
static auto render_settings_vrtxR_min = 0.f;
static auto render_settings_vrtxR_max = 1.f;
ParameterManager::instance().newParameter("render_settings.vrtxR", float{0.586f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxR_min, render_settings_vrtxR_max},
	});
static auto render_settings_camera_fov_min = 0.f;
static auto render_settings_camera_fov_max = 256.f;
ParameterManager::instance().newParameter("render_settings.camera_fov", float{96.f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_camera_fov_min, render_settings_camera_fov_max},
	});
static auto render_settings_vrtxWMax_min = 0.f;
static auto render_settings_vrtxWMax_max = 4.f;
ParameterManager::instance().newParameter("render_settings.vrtxWMax", float{2.f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxWMax_min, render_settings_vrtxWMax_max},
	});
static auto render_settings_vrtxBounces_min = 0;
static auto render_settings_vrtxBounces_max = 64;
ParameterManager::instance().newParameter("render_settings.vrtxBounces", int32_t{5},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxBounces_min, render_settings_vrtxBounces_max},
	});
static auto render_settings_auxScale_min = 0.25f;
static auto render_settings_auxScale_max = 16.f;
ParameterManager::instance().newParameter("render_settings.auxScale", float{1.f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_auxScale_min, render_settings_auxScale_max},
	});
static auto render_settings_vrtxIOR_min = 1.f;
static auto render_settings_vrtxIOR_max = 5.f;
ParameterManager::instance().newParameter("render_settings.vrtxIOR", float{1.3f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_vrtxIOR_min, render_settings_vrtxIOR_max},
	});
static auto render_settings_renderSteps_min = 1;
static auto render_settings_renderSteps_max = 50;
ParameterManager::instance().newParameter("render_settings.renderSteps", int32_t{25},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_renderSteps_min, render_settings_renderSteps_max},
	});
static auto render_settings_internalLimit_min = 0.f;
static auto render_settings_internalLimit_max = 64.f;
ParameterManager::instance().newParameter("render_settings.internalLimit", float{40.f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_internalLimit_min, render_settings_internalLimit_max},
	});
ParameterManager::instance().newParameter("render_settings.auxCellCount", int32_t{-1},{
		.description = ".",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
static auto render_settings_axesRender_min = 0;
static auto render_settings_axesRender_max = 1;
ParameterManager::instance().newParameter("render_settings.axesRender", int32_t{1},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_axesRender_min, render_settings_axesRender_max},
	});
static auto render_settings_boundsRender_min = 0;
static auto render_settings_boundsRender_max = 1;
ParameterManager::instance().newParameter("render_settings.boundsRender", int32_t{1},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_boundsRender_min, render_settings_boundsRender_max},
	});
static auto render_settings_floorRender_min = 0;
static auto render_settings_floorRender_max = 1;
ParameterManager::instance().newParameter("render_settings.floorRender", int32_t{1},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_floorRender_min, render_settings_floorRender_max},
	});
static auto render_settings_axesScale_min = 0.f;
static auto render_settings_axesScale_max = 64.f;
ParameterManager::instance().newParameter("render_settings.axesScale", float{1.f},{
		.description = ".",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_axesScale_min, render_settings_axesScale_max},
	});
static auto render_settings_render_clamp_min = -1.f;
static auto render_settings_render_clamp_max = 1.f;
ParameterManager::instance().newParameter("render_settings.render_clamp", float3{0.f,0.f,0.f},{
		.description = "Describes a clipping plane for the simulaltion relative to the simulations AABB used for rendering. Negative values flip what side is clipped. Useful value: {0.f,0.f,0.f}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_render_clamp_min, render_settings_render_clamp_max},
	});
static auto render_settings_camera_position_min = -1000.f;
static auto render_settings_camera_position_max = 1000.f;
ParameterManager::instance().newParameter("render_settings.camera_position", float3{125, 0, -50},{
		.description = "Camera position in world coordinates. Useful value: {125 0 -50}",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat3<SI::m>)},
		.range = Range{render_settings_camera_position_min, render_settings_camera_position_max},
	});
static auto render_settings_camera_angle_min = -360.f;
static auto render_settings_camera_angle_max = 360.f;
ParameterManager::instance().newParameter("render_settings.camera_angle", float3{-90, 0, 90},{
		.description = "Angle of the camera (based on houdinis representation of cameras). Useful value: {-90 0 90}.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{render_settings_camera_angle_min, render_settings_camera_angle_max},
	});
ParameterManager::instance().newParameter("render_settings.camera_resolution", float2{1920, 1080},{
		.description = "Resolution of the render window. Currently not used.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("render_settings.camera_fps", float{60.f},{
		.description = "FPS target for the export from the openGL viewport into a file.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("render_settings.gl_file", std::string{"gl.mp4"},{
		.description = "File the openGL viewport should be rendered into.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("resort.auxCells", int{0},{
		.description = "Internal value to keep track of the resorting algorithm being used, deprecated.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("resort.auxCollisions", int{0},{
		.description = "Internal value to keep track of the resorting algorithm being used, deprecated.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("resort.resort_algorithm", int{0},{
		.description = "Internal value to keep track of the resorting algorithm being used, deprecated.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("resort.valid_cells", int{0},{
		.description = "Value that represents the number of occupied hash cells (on all hash levels)",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("resort.zOrderScale", float{1.f},{
		.description = "Value that represents the number of occupied hash cells (on all hash levels)",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("resort.collision_cells", int{0},{
		.description = "Value that represents the number of hash cells that contain collisions (on all hash levels)",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("resort.occupiedCells", std::vector<int32_t>{0},{
		.description = "Value that represents the number of hash cells that contain collisions (on all hash levels)",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("rigid_volumes.mesh_resolution", int{20},{
		.description = "",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
static auto rigid_volumes_gamma_min = 0.000f;
static auto rigid_volumes_gamma_max = 5.f;
ParameterManager::instance().newParameter("rigid_volumes.gamma", float{0.7f},{
		.description = "",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{rigid_volumes_gamma_min, rigid_volumes_gamma_max},
	});
static auto rigid_volumes_beta_min = 0.000f;
static auto rigid_volumes_beta_max = 5.f;
ParameterManager::instance().newParameter("rigid_volumes.beta", float{0.1f},{
		.description = "",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{rigid_volumes_beta_min, rigid_volumes_beta_max},
	});
ParameterManager::instance().newParameter("rigid_volumes.volume", std::vector<rigidVolume>{},{
		.description = "Complex parameter that describes a volume that should be loaded from a file and emitted as rigid particles once.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("rtxScene.sphere", std::vector<rtxSphere>{},{
		.description = "Complex parameter that describes a volume that should be loaded from a file and emitted as particles once.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("rtxScene.box", std::vector<rtxBox>{},{
		.description = "Complex parameter that describes a volume that should be loaded from a file and emitted as particles once.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
static auto simulation_settings_external_force_min = -10.f;
static auto simulation_settings_external_force_max = 10.f;
ParameterManager::instance().newParameter("simulation_settings.external_force", float4{0.f,0.f,-9.81f,0.f},{
		.description = "External force being applied to the particles, e.g. gravity. Useful value: {0,0,-9.81,0}",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat4<SI::acceleration>)},
		.range = Range{simulation_settings_external_force_min, simulation_settings_external_force_max},
	});
static auto simulation_settings_timestep_min_min = 0.001f;
static auto simulation_settings_timestep_min_max = 0.01f;
ParameterManager::instance().newParameter("simulation_settings.timestep_min", float{0.001f},{
		.description = "Lowest allowed timestep for the simulation. Useful value: 0.001f",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::s>)},
		.range = Range{simulation_settings_timestep_min_min, simulation_settings_timestep_min_max},
	});
static auto simulation_settings_timestep_max_min = 0.001f;
static auto simulation_settings_timestep_max_max = 0.01f;
ParameterManager::instance().newParameter("simulation_settings.timestep_max", float{0.01f},{
		.description = "Largest allowed timestep for the simulation. Useful value: 0.01f",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::s>)},
		.range = Range{simulation_settings_timestep_max_min, simulation_settings_timestep_max_max},
	});
static auto simulation_settings_boundaryDampening_min = 0.0f;
static auto simulation_settings_boundaryDampening_max = 1.0f;
ParameterManager::instance().newParameter("simulation_settings.boundaryDampening", float{0.97f},{
		.description = "Dampening applied to particles upon impact of the boundary, currently effectively forced to 1.",
		.constant = false,
		.hidden = true,
		.alternativeTypes{},
		.range = Range{simulation_settings_boundaryDampening_min, simulation_settings_boundaryDampening_max},
	});
static auto simulation_settings_LUTOffset_min = -2.285391f;
static auto simulation_settings_LUTOffset_max = 2.285391f;
ParameterManager::instance().newParameter("simulation_settings.LUTOffset", float{0.f},{
		.description = "Path to an .obj file used to initialize the Simulation domain.",
		.constant = false,
		.hidden = true,
		.alternativeTypes{},
		.range = Range{simulation_settings_LUTOffset_min, simulation_settings_LUTOffset_max},
	});
ParameterManager::instance().newParameter("simulation_settings.boundaryObject", std::string{""},{
		.description = "Path to an .obj file used to initialize the Simulation domain.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("simulation_settings.domainWalls", std::string{"x+-y+-z+-"},{
		.description = "String used to create boundary walls around the domain described by the boundaryObject. Useful value: x+-y+-z+-.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("simulation_settings.neighborlimit", int32_t{150u},{
		.description = "Maximum number of neighborhood entries for a particle.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("simulation_settings.dumpFile", std::string{"simulation.dump"},{
		.description = "Internal value that represents the used launch configuration, not visible in the simulation as it is an enum.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("simulation_settings.maxNumptcls", int32_t{1000000u},{
		.description = "Represents the maximum number of particles allowed in the simulation due to memory allocations.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("simulation_settings.hash_entries", uint32_t{UINT_MAX},{
		.description = "Represents the size of the hash tables for compact hashing. Important: Choose a prime number.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("simulation_settings.mlm_schemes", uint32_t{UINT_MAX},{
		.description = "Represents the levels of hash tables beign used for the MLM algorithm. Useful value: 3.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("simulation_settings.deviceRegex", std::string{""},{
		.description = "String used to create boundary walls around the domain described by the boundaryObject. Useful value: x+-y+-z+-.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("simulation_settings.hostRegex", std::string{""},{
		.description = "String used to create boundary walls around the domain described by the boundaryObject. Useful value: x+-y+-z+-.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("simulation_settings.debugRegex", std::string{""},{
		.description = "String used to create boundary walls around the domain described by the boundaryObject. Useful value: x+-y+-z+-.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("simulation_settings.densitySteps", int32_t{10u},{
		.description = "Represents the maximum number of particles allowed in the simulation due to memory allocations.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("support.support_current_iteration", uint32_t{0},{
		.description = "Number of iterations required to constrain the support.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("support.adjusted_particles", int32_t{0},{
		.description = "Number of particles with an adjusted support radius in the last iteration.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("support.omega", float{0.97f},{
		.description = "Scaling factor for changing the support radius back to the default value. Useful value: 0.97f.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("support.target_neighbors", int32_t{0},{
		.description = "Represents the ideal number of neighbors for a particle. Based on the kernel being used.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("support.support_leeway", int32_t{0},{
		.description = "Difference between the number of neighbor entries per particle and the target number of neighbor entries.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("support.overhead_size", int32_t{0},{
		.description = "Represents the overhead size used to temporarily store neighbor entries.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("support.error_factor", int32_t{3},{
		.description = "Used as an offset to create the actual neighbor entries as forcing values to be as small as the ideal number can be problematic.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("surfaceDistance.surface_levelLimit", float{-20.f},{
		.description = "Maximum distance to the surface that is calculate by the surface distance function. Should be atleast -20.f or smaller.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::m>)},
	});
ParameterManager::instance().newParameter("surfaceDistance.surface_neighborLimit", int32_t{40},{
		.description = "Particles with more than this number of neighbors will be marked as interior particles. Useful value: 40.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("surfaceDistance.surface_phiMin", float{0.f},{
		.description = "Value that stores the closest distance of a fluid particle to the surface.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{typeid(uFloat<SI::m>)},
	});
ParameterManager::instance().newParameter("surfaceDistance.surface_phiChange", float{0.f},{
		.description = "Value that stores the number of particles changed in the last distance iteration.",
		.constant = true,
		.hidden = false,
		.alternativeTypes{},
	});
ParameterManager::instance().newParameter("surfaceDistance.surface_distanceFieldDistances", float3{0.f,0.f,1.5f},{
		.description = "Particles closer than this distance to a boundary will be marked as interior particles to avoid instabilities due to bad boundary handling. Useful value: 4.f,4.f,4.f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{typeid(uFloat3<SI::m>)},
	});
ParameterManager::instance().newParameter("surfaceDistance.surface_iterations", int32_t{0},{
		.description = "Particles closer than this distance to a boundary will be marked as interior particles to avoid instabilities due to bad boundary handling. Useful value: 4.f,4.f,4.f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
	});
static auto vorticitySettings_intertiaInverse_min = 0.000f;
static auto vorticitySettings_intertiaInverse_max = 1.f;
ParameterManager::instance().newParameter("vorticitySettings.intertiaInverse", float{0.5f},{
		.description = "Parameter used for the micropolar SPH model. Useful value: 0.5f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{vorticitySettings_intertiaInverse_min, vorticitySettings_intertiaInverse_max},
	});
static auto vorticitySettings_viscosityOmega_min = 0.000f;
static auto vorticitySettings_viscosityOmega_max = 1.f;
ParameterManager::instance().newParameter("vorticitySettings.viscosityOmega", float{0.1f},{
		.description = "Parameter used for the micropolar SPH model. Useful value: 0.1f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{vorticitySettings_viscosityOmega_min, vorticitySettings_viscosityOmega_max},
	});
static auto vorticitySettings_vorticityCoeff_min = 0.000f;
static auto vorticitySettings_vorticityCoeff_max = 1.f;
ParameterManager::instance().newParameter("vorticitySettings.vorticityCoeff", float{0.05f},{
		.description = "Parameter used for the micropolar SPH model. Useful value: 0.05f.",
		.constant = false,
		.hidden = false,
		.alternativeTypes{},
		.range = Range{vorticitySettings_vorticityCoeff_min, vorticitySettings_vorticityCoeff_max},
	});
ParameterManager::instance().addEncoder(typeid(boundaryVolume), [](const detail::iAny& any) {
	const auto& var = boost::any_cast<const boundaryVolume&>(any);
	auto node  = YAML::Node();
	node["fileName"] = callEncoder(std::string, var.fileName);
	node["density"] = callEncoder(float, var.density);
	node["old_density"] = callEncoder(float, var.old_density);
	node["position"] = callEncoder(float3, var.position);
	node["velocity"] = callEncoder(float3, var.velocity);
	node["angularVelocity"] = callEncoder(float4, var.angularVelocity);
	node["angle"] = callEncoder(float3, var.angle);
	node["kind"] = callEncoder(int32_t, var.kind);
	node["animationPath"] = callEncoder(std::string, var.animationPath);
	return node;
});
ParameterManager::instance().addEncoder(typeid(inletVolume), [](const detail::iAny& any) {
	const auto& var = boost::any_cast<const inletVolume&>(any);
	auto node  = YAML::Node();
	node["fileName"] = callEncoder(std::string, var.fileName);
	node["particles_emitted"] = callEncoder(int32_t, var.particles_emitted);
	node["duration"] = callEncoder(float, var.duration);
	node["delay"] = callEncoder(float, var.delay);
	node["inlet_radius"] = callEncoder(float, var.inlet_radius);
	node["emitter_velocity"] = callEncoder(float4, var.emitter_velocity);
	return node;
});
ParameterManager::instance().addEncoder(typeid(movingPlane), [](const detail::iAny& any) {
	const auto& var = boost::any_cast<const movingPlane&>(any);
	auto node  = YAML::Node();
	node["plane_position"] = callEncoder(float3, var.plane_position);
	node["plane_normal"] = callEncoder(float3, var.plane_normal);
	node["plane_direction"] = callEncoder(float3, var.plane_direction);
	node["duration"] = callEncoder(float, var.duration);
	node["magnitude"] = callEncoder(float, var.magnitude);
	node["frequency"] = callEncoder(float, var.frequency);
	node["index"] = callEncoder(int32_t, var.index);
	return node;
});
ParameterManager::instance().addEncoder(typeid(outletVolume), [](const detail::iAny& any) {
	const auto& var = boost::any_cast<const outletVolume&>(any);
	auto node  = YAML::Node();
	node["fileName"] = callEncoder(std::string, var.fileName);
	node["duration"] = callEncoder(float, var.duration);
	node["delay"] = callEncoder(float, var.delay);
	node["flowRate"] = callEncoder(float, var.flowRate);
	return node;
});
ParameterManager::instance().addEncoder(typeid(particleVolume), [](const detail::iAny& any) {
	const auto& var = boost::any_cast<const particleVolume&>(any);
	auto node  = YAML::Node();
	node["fileName"] = callEncoder(std::string, var.fileName);
	node["kind"] = callEncoder(std::string, var.kind);
	node["concentration"] = callEncoder(float, var.concentration);
	node["timeToEmit"] = callEncoder(float, var.timeToEmit);
	node["scale"] = callEncoder(float3, var.scale);
	node["shift"] = callEncoder(float3, var.shift);
	node["velocity"] = callEncoder(float3, var.velocity);
	return node;
});
ParameterManager::instance().addEncoder(typeid(rigidVolume), [](const detail::iAny& any) {
	const auto& var = boost::any_cast<const rigidVolume&>(any);
	auto node  = YAML::Node();
	node["fileName"] = callEncoder(std::string, var.fileName);
	node["kind"] = callEncoder(std::string, var.kind);
	node["density"] = callEncoder(float, var.density);
	node["shift"] = callEncoder(float3, var.shift);
	node["concentration"] = callEncoder(float, var.concentration);
	node["sampling"] = callEncoder(float, var.sampling);
	node["init_velocity"] = callEncoder(float3, var.init_velocity);
	node["scale"] = callEncoder(float3, var.scale);
	node["timeToEmit"] = callEncoder(float, var.timeToEmit);
	return node;
});
ParameterManager::instance().addEncoder(typeid(rtxSphere), [](const detail::iAny& any) {
	const auto& var = boost::any_cast<const rtxSphere&>(any);
	auto node  = YAML::Node();
	node["radius"] = callEncoder(float, var.radius);
	node["position"] = callEncoder(float3, var.position);
	node["emission"] = callEncoder(float3, var.emission);
	node["color"] = callEncoder(float3, var.color);
	node["refl_t"] = callEncoder(int32_t, var.refl_t);
	return node;
});
ParameterManager::instance().addEncoder(typeid(rtxBox), [](const detail::iAny& any) {
	const auto& var = boost::any_cast<const rtxBox&>(any);
	auto node  = YAML::Node();
	node["maxPosition"] = callEncoder(std::string, var.maxPosition);
	node["minPosition"] = callEncoder(std::string, var.minPosition);
	node["emission"] = callEncoder(float3, var.emission);
	node["color"] = callEncoder(float3, var.color);
	node["refl_t"] = callEncoder(int32_t, var.refl_t);
	return node;
});
ParameterManager::instance().addDecoder(typeid(boundaryVolume), [](const YAML::Node& node) {
	 boundaryVolume var;
	var.fileName = callDecoder(std::string, node["fileName"], var.fileName);
	var.density = callDecoder(float, node["density"], var.density);
	var.old_density = callDecoder(float, node["old_density"], var.old_density);
	var.position = callDecoder(float3, node["position"], var.position);
	var.velocity = callDecoder(float3, node["velocity"], var.velocity);
	var.angularVelocity = callDecoder(float4, node["angularVelocity"], var.angularVelocity);
	var.angle = callDecoder(float3, node["angle"], var.angle);
	var.kind = callDecoder(int32_t, node["kind"], var.kind);
	var.animationPath = callDecoder(std::string, node["animationPath"], var.animationPath);
	return detail::iAny(var);
});
ParameterManager::instance().addDecoder(typeid(inletVolume), [](const YAML::Node& node) {
	 inletVolume var;
	var.fileName = callDecoder(std::string, node["fileName"], var.fileName);
	var.particles_emitted = callDecoder(int32_t, node["particles_emitted"], var.particles_emitted);
	var.duration = callDecoder(float, node["duration"], var.duration);
	var.delay = callDecoder(float, node["delay"], var.delay);
	var.inlet_radius = callDecoder(float, node["inlet_radius"], var.inlet_radius);
	var.emitter_velocity = callDecoder(float4, node["emitter_velocity"], var.emitter_velocity);
	return detail::iAny(var);
});
ParameterManager::instance().addDecoder(typeid(movingPlane), [](const YAML::Node& node) {
	 movingPlane var;
	var.plane_position = callDecoder(float3, node["plane_position"], var.plane_position);
	var.plane_normal = callDecoder(float3, node["plane_normal"], var.plane_normal);
	var.plane_direction = callDecoder(float3, node["plane_direction"], var.plane_direction);
	var.duration = callDecoder(float, node["duration"], var.duration);
	var.magnitude = callDecoder(float, node["magnitude"], var.magnitude);
	var.frequency = callDecoder(float, node["frequency"], var.frequency);
	var.index = callDecoder(int32_t, node["index"], var.index);
	return detail::iAny(var);
});
ParameterManager::instance().addDecoder(typeid(outletVolume), [](const YAML::Node& node) {
	 outletVolume var;
	var.fileName = callDecoder(std::string, node["fileName"], var.fileName);
	var.duration = callDecoder(float, node["duration"], var.duration);
	var.delay = callDecoder(float, node["delay"], var.delay);
	var.flowRate = callDecoder(float, node["flowRate"], var.flowRate);
	return detail::iAny(var);
});
ParameterManager::instance().addDecoder(typeid(particleVolume), [](const YAML::Node& node) {
	 particleVolume var;
	var.fileName = callDecoder(std::string, node["fileName"], var.fileName);
	var.kind = callDecoder(std::string, node["kind"], var.kind);
	var.concentration = callDecoder(float, node["concentration"], var.concentration);
	var.timeToEmit = callDecoder(float, node["timeToEmit"], var.timeToEmit);
	var.scale = callDecoder(float3, node["scale"], var.scale);
	var.shift = callDecoder(float3, node["shift"], var.shift);
	var.velocity = callDecoder(float3, node["velocity"], var.velocity);
	return detail::iAny(var);
});
ParameterManager::instance().addDecoder(typeid(rigidVolume), [](const YAML::Node& node) {
	 rigidVolume var;
	var.fileName = callDecoder(std::string, node["fileName"], var.fileName);
	var.kind = callDecoder(std::string, node["kind"], var.kind);
	var.density = callDecoder(float, node["density"], var.density);
	var.shift = callDecoder(float3, node["shift"], var.shift);
	var.concentration = callDecoder(float, node["concentration"], var.concentration);
	var.sampling = callDecoder(float, node["sampling"], var.sampling);
	var.init_velocity = callDecoder(float3, node["init_velocity"], var.init_velocity);
	var.scale = callDecoder(float3, node["scale"], var.scale);
	var.timeToEmit = callDecoder(float, node["timeToEmit"], var.timeToEmit);
	return detail::iAny(var);
});
ParameterManager::instance().addDecoder(typeid(rtxSphere), [](const YAML::Node& node) {
	 rtxSphere var;
	var.radius = callDecoder(float, node["radius"], var.radius);
	var.position = callDecoder(float3, node["position"], var.position);
	var.emission = callDecoder(float3, node["emission"], var.emission);
	var.color = callDecoder(float3, node["color"], var.color);
	var.refl_t = callDecoder(int32_t, node["refl_t"], var.refl_t);
	return detail::iAny(var);
});
ParameterManager::instance().addDecoder(typeid(rtxBox), [](const YAML::Node& node) {
	 rtxBox var;
	var.maxPosition = callDecoder(std::string, node["maxPosition"], var.maxPosition);
	var.minPosition = callDecoder(std::string, node["minPosition"], var.minPosition);
	var.emission = callDecoder(float3, node["emission"], var.emission);
	var.color = callDecoder(float3, node["color"], var.color);
	var.refl_t = callDecoder(int32_t, node["refl_t"], var.refl_t);
	return detail::iAny(var);
});
ParameterManager::instance().addUifunction(typeid(boundaryVolume), [](Parameter& parameter) {
	boundaryVolume& val = boost::any_cast<boundaryVolume&>(parameter.param.val.value());
	if (parameter.properties.hidden) return;
	ImGui::Text(parameter.identifier.c_str());
	ImGui::PushID(parameter.identifier.c_str());
	callUI(std::string, val.fileName, ".fileName");
	callUI(float, val.density, ".density");
	callUI(float, val.old_density, ".old_density");
	callUI(float3, val.position, ".position");
	callUI(float3, val.velocity, ".velocity");
	callUI(float4, val.angularVelocity, ".angularVelocity");
	callUI(float3, val.angle, ".angle");
	callUI(int32_t, val.kind, ".kind");
	callUI(std::string, val.animationPath, ".animationPath");
	ImGui::PopID();
});
	customVector(boundaryVolume);
ParameterManager::instance().addUifunction(typeid(inletVolume), [](Parameter& parameter) {
	inletVolume& val = boost::any_cast<inletVolume&>(parameter.param.val.value());
	if (parameter.properties.hidden) return;
	ImGui::Text(parameter.identifier.c_str());
	ImGui::PushID(parameter.identifier.c_str());
	callUI(std::string, val.fileName, ".fileName");
	callUI(int32_t, val.particles_emitted, ".particles_emitted");
	callUI(float, val.duration, ".duration");
	callUI(float, val.delay, ".delay");
	callUI(float, val.inlet_radius, ".inlet_radius");
	callUI(float4, val.emitter_velocity, ".emitter_velocity");
	ImGui::PopID();
});
	customVector(inletVolume);
ParameterManager::instance().addUifunction(typeid(movingPlane), [](Parameter& parameter) {
	movingPlane& val = boost::any_cast<movingPlane&>(parameter.param.val.value());
	if (parameter.properties.hidden) return;
	ImGui::Text(parameter.identifier.c_str());
	ImGui::PushID(parameter.identifier.c_str());
	callUI(float3, val.plane_position, ".plane_position");
	callUI(float3, val.plane_normal, ".plane_normal");
	callUI(float3, val.plane_direction, ".plane_direction");
	callUI(float, val.duration, ".duration");
	callUI(float, val.magnitude, ".magnitude");
	callUI(float, val.frequency, ".frequency");
	callUI(int32_t, val.index, ".index");
	ImGui::PopID();
});
	customVector(movingPlane);
ParameterManager::instance().addUifunction(typeid(outletVolume), [](Parameter& parameter) {
	outletVolume& val = boost::any_cast<outletVolume&>(parameter.param.val.value());
	if (parameter.properties.hidden) return;
	ImGui::Text(parameter.identifier.c_str());
	ImGui::PushID(parameter.identifier.c_str());
	callUI(std::string, val.fileName, ".fileName");
	callUI(float, val.duration, ".duration");
	callUI(float, val.delay, ".delay");
	callUI(float, val.flowRate, ".flowRate");
	ImGui::PopID();
});
	customVector(outletVolume);
ParameterManager::instance().addUifunction(typeid(particleVolume), [](Parameter& parameter) {
	particleVolume& val = boost::any_cast<particleVolume&>(parameter.param.val.value());
	if (parameter.properties.hidden) return;
	ImGui::Text(parameter.identifier.c_str());
	ImGui::PushID(parameter.identifier.c_str());
	callUI(std::string, val.fileName, ".fileName");
	callUI(std::string, val.kind, ".kind");
	callUI(float, val.concentration, ".concentration");
	callUI(float, val.timeToEmit, ".timeToEmit");
	callUI(float3, val.scale, ".scale");
	callUI(float3, val.shift, ".shift");
	callUI(float3, val.velocity, ".velocity");
	ImGui::PopID();
});
	customVector(particleVolume);
ParameterManager::instance().addUifunction(typeid(rigidVolume), [](Parameter& parameter) {
	rigidVolume& val = boost::any_cast<rigidVolume&>(parameter.param.val.value());
	if (parameter.properties.hidden) return;
	ImGui::Text(parameter.identifier.c_str());
	ImGui::PushID(parameter.identifier.c_str());
	callUI(std::string, val.fileName, ".fileName");
	callUI(std::string, val.kind, ".kind");
	callUI(float, val.density, ".density");
	callUI(float3, val.shift, ".shift");
	callUI(float, val.concentration, ".concentration");
	callUI(float, val.sampling, ".sampling");
	callUI(float3, val.init_velocity, ".init_velocity");
	callUI(float3, val.scale, ".scale");
	callUI(float, val.timeToEmit, ".timeToEmit");
	ImGui::PopID();
});
	customVector(rigidVolume);
ParameterManager::instance().addUifunction(typeid(rtxSphere), [](Parameter& parameter) {
	rtxSphere& val = boost::any_cast<rtxSphere&>(parameter.param.val.value());
	if (parameter.properties.hidden) return;
	ImGui::Text(parameter.identifier.c_str());
	ImGui::PushID(parameter.identifier.c_str());
	callUI(float, val.radius, ".radius");
	callUI(float3, val.position, ".position");
	callUI(float3, val.emission, ".emission");
	callUI(float3, val.color, ".color");
	callUI(int32_t, val.refl_t, ".refl_t");
	ImGui::PopID();
});
	customVector(rtxSphere);
ParameterManager::instance().addUifunction(typeid(rtxBox), [](Parameter& parameter) {
	rtxBox& val = boost::any_cast<rtxBox&>(parameter.param.val.value());
	if (parameter.properties.hidden) return;
	ImGui::Text(parameter.identifier.c_str());
	ImGui::PushID(parameter.identifier.c_str());
	callUI(std::string, val.maxPosition, ".maxPosition");
	callUI(std::string, val.minPosition, ".minPosition");
	callUI(float3, val.emission, ".emission");
	callUI(float3, val.color, ".color");
	callUI(int32_t, val.refl_t, ".refl_t");
	ImGui::PopID();
});
	customVector(rtxBox);

}	template<> typename getType<parameters::adaptive,parameters::adaptive::adaptivityScaling>::type& get<parameters::adaptive::adaptivityScaling>(){
		return ParameterManager::instance().get<float>("adaptive.adaptivityScaling");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::adaptivityThreshold>::type& get<parameters::adaptive::adaptivityThreshold>(){
		return ParameterManager::instance().get<float>("adaptive.adaptivityThreshold");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::adaptivityGamma>::type& get<parameters::adaptive::adaptivityGamma>(){
		return ParameterManager::instance().get<float>("adaptive.adaptivityGamma");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::resolution>::type& get<parameters::adaptive::resolution>(){
		return ParameterManager::instance().get<float>("adaptive.resolution");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::useVolume>::type& get<parameters::adaptive::useVolume>(){
		return ParameterManager::instance().get<int32_t>("adaptive.useVolume");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::minVolume>::type& get<parameters::adaptive::minVolume>(){
		return ParameterManager::instance().get<float>("adaptive.minVolume");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::detailedAdaptiveStatistics>::type& get<parameters::adaptive::detailedAdaptiveStatistics>(){
		return ParameterManager::instance().get<int32_t>("adaptive.detailedAdaptiveStatistics");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::ratio>::type& get<parameters::adaptive::ratio>(){
		return ParameterManager::instance().get<float>("adaptive.ratio");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::blendSteps>::type& get<parameters::adaptive::blendSteps>(){
		return ParameterManager::instance().get<float>("adaptive.blendSteps");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::delay>::type& get<parameters::adaptive::delay>(){
		return ParameterManager::instance().get<float>("adaptive.delay");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::splitPtcls>::type& get<parameters::adaptive::splitPtcls>(){
		return ParameterManager::instance().get<std::vector<int32_t>>("adaptive.splitPtcls");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::blendedPtcls>::type& get<parameters::adaptive::blendedPtcls>(){
		return ParameterManager::instance().get<int32_t>("adaptive.blendedPtcls");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::mergedPtcls>::type& get<parameters::adaptive::mergedPtcls>(){
		return ParameterManager::instance().get<std::vector<int32_t>>("adaptive.mergedPtcls");
	}
	template<> typename getType<parameters::adaptive,parameters::adaptive::sharedPtcls>::type& get<parameters::adaptive::sharedPtcls>(){
		return ParameterManager::instance().get<std::vector<int32_t>>("adaptive.sharedPtcls");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::adaptive ident){
		if(ident == parameters::adaptive::adaptivityScaling) return std::make_pair(std::string("adaptive"), std::string("adaptivityScaling"));
		if(ident == parameters::adaptive::adaptivityThreshold) return std::make_pair(std::string("adaptive"), std::string("adaptivityThreshold"));
		if(ident == parameters::adaptive::adaptivityGamma) return std::make_pair(std::string("adaptive"), std::string("adaptivityGamma"));
		if(ident == parameters::adaptive::resolution) return std::make_pair(std::string("adaptive"), std::string("resolution"));
		if(ident == parameters::adaptive::useVolume) return std::make_pair(std::string("adaptive"), std::string("useVolume"));
		if(ident == parameters::adaptive::minVolume) return std::make_pair(std::string("adaptive"), std::string("minVolume"));
		if(ident == parameters::adaptive::detailedAdaptiveStatistics) return std::make_pair(std::string("adaptive"), std::string("detailedAdaptiveStatistics"));
		if(ident == parameters::adaptive::ratio) return std::make_pair(std::string("adaptive"), std::string("ratio"));
		if(ident == parameters::adaptive::blendSteps) return std::make_pair(std::string("adaptive"), std::string("blendSteps"));
		if(ident == parameters::adaptive::delay) return std::make_pair(std::string("adaptive"), std::string("delay"));
		if(ident == parameters::adaptive::splitPtcls) return std::make_pair(std::string("adaptive"), std::string("splitPtcls"));
		if(ident == parameters::adaptive::blendedPtcls) return std::make_pair(std::string("adaptive"), std::string("blendedPtcls"));
		if(ident == parameters::adaptive::mergedPtcls) return std::make_pair(std::string("adaptive"), std::string("mergedPtcls"));
		if(ident == parameters::adaptive::sharedPtcls) return std::make_pair(std::string("adaptive"), std::string("sharedPtcls"));
	}
	template<> typename getType<parameters::alembic,parameters::alembic::file_name>::type& get<parameters::alembic::file_name>(){
		return ParameterManager::instance().get<std::string>("alembic.file_name");
	}
	template<> typename getType<parameters::alembic,parameters::alembic::fps>::type& get<parameters::alembic::fps>(){
		return ParameterManager::instance().get<int32_t>("alembic.fps");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::alembic ident){
		if(ident == parameters::alembic::file_name) return std::make_pair(std::string("alembic"), std::string("file_name"));
		if(ident == parameters::alembic::fps) return std::make_pair(std::string("alembic"), std::string("fps"));
	}
	template<> typename getType<parameters::boundary_volumes,parameters::boundary_volumes::volumeBoundaryCounter>::type& get<parameters::boundary_volumes::volumeBoundaryCounter>(){
		return ParameterManager::instance().get<int32_t>("boundary_volumes.volumeBoundaryCounter");
	}
	template<> typename getType<parameters::boundary_volumes,parameters::boundary_volumes::volume>::type& get<parameters::boundary_volumes::volume>(){
		return ParameterManager::instance().get<std::vector<boundaryVolume>>("boundary_volumes.volume");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::boundary_volumes ident){
		if(ident == parameters::boundary_volumes::volumeBoundaryCounter) return std::make_pair(std::string("boundary_volumes"), std::string("volumeBoundaryCounter"));
		if(ident == parameters::boundary_volumes::volume) return std::make_pair(std::string("boundary_volumes"), std::string("volume"));
	}
	template<> typename getType<parameters::color_map,parameters::color_map::transfer_mode>::type& get<parameters::color_map::transfer_mode>(){
		return ParameterManager::instance().get<std::string>("color_map.transfer_mode");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::mapping_mode>::type& get<parameters::color_map::mapping_mode>(){
		return ParameterManager::instance().get<std::string>("color_map.mapping_mode");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::vectorMode>::type& get<parameters::color_map::vectorMode>(){
		return ParameterManager::instance().get<std::string>("color_map.vectorMode");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::visualizeDirection>::type& get<parameters::color_map::visualizeDirection>(){
		return ParameterManager::instance().get<int32_t>("color_map.visualizeDirection");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::vectorScale>::type& get<parameters::color_map::vectorScale>(){
		return ParameterManager::instance().get<float>("color_map.vectorScale");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::vectorScaling>::type& get<parameters::color_map::vectorScaling>(){
		return ParameterManager::instance().get<int32_t>("color_map.vectorScaling");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::min>::type& get<parameters::color_map::min>(){
		return ParameterManager::instance().get<float>("color_map.min");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::max>::type& get<parameters::color_map::max>(){
		return ParameterManager::instance().get<float>("color_map.max");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::transfer_fn>::type& get<parameters::color_map::transfer_fn>(){
		return ParameterManager::instance().get<int32_t>("color_map.transfer_fn");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::pruneVoxel>::type& get<parameters::color_map::pruneVoxel>(){
		return ParameterManager::instance().get<int32_t>("color_map.pruneVoxel");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::mapping_fn>::type& get<parameters::color_map::mapping_fn>(){
		return ParameterManager::instance().get<int32_t>("color_map.mapping_fn");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::autoScaling>::type& get<parameters::color_map::autoScaling>(){
		return ParameterManager::instance().get<int>("color_map.autoScaling");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::map_flipped>::type& get<parameters::color_map::map_flipped>(){
		return ParameterManager::instance().get<int>("color_map.map_flipped");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::buffer>::type& get<parameters::color_map::buffer>(){
		return ParameterManager::instance().get<std::string>("color_map.buffer");
	}
	template<> typename getType<parameters::color_map,parameters::color_map::map>::type& get<parameters::color_map::map>(){
		return ParameterManager::instance().get<std::string>("color_map.map");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::color_map ident){
		if(ident == parameters::color_map::transfer_mode) return std::make_pair(std::string("color_map"), std::string("transfer_mode"));
		if(ident == parameters::color_map::mapping_mode) return std::make_pair(std::string("color_map"), std::string("mapping_mode"));
		if(ident == parameters::color_map::vectorMode) return std::make_pair(std::string("color_map"), std::string("vectorMode"));
		if(ident == parameters::color_map::visualizeDirection) return std::make_pair(std::string("color_map"), std::string("visualizeDirection"));
		if(ident == parameters::color_map::vectorScale) return std::make_pair(std::string("color_map"), std::string("vectorScale"));
		if(ident == parameters::color_map::vectorScaling) return std::make_pair(std::string("color_map"), std::string("vectorScaling"));
		if(ident == parameters::color_map::min) return std::make_pair(std::string("color_map"), std::string("min"));
		if(ident == parameters::color_map::max) return std::make_pair(std::string("color_map"), std::string("max"));
		if(ident == parameters::color_map::transfer_fn) return std::make_pair(std::string("color_map"), std::string("transfer_fn"));
		if(ident == parameters::color_map::pruneVoxel) return std::make_pair(std::string("color_map"), std::string("pruneVoxel"));
		if(ident == parameters::color_map::mapping_fn) return std::make_pair(std::string("color_map"), std::string("mapping_fn"));
		if(ident == parameters::color_map::autoScaling) return std::make_pair(std::string("color_map"), std::string("autoScaling"));
		if(ident == parameters::color_map::map_flipped) return std::make_pair(std::string("color_map"), std::string("map_flipped"));
		if(ident == parameters::color_map::buffer) return std::make_pair(std::string("color_map"), std::string("buffer"));
		if(ident == parameters::color_map::map) return std::make_pair(std::string("color_map"), std::string("map"));
	}
	template<> typename getType<parameters::dfsph_settings,parameters::dfsph_settings::densityError>::type& get<parameters::dfsph_settings::densityError>(){
		return ParameterManager::instance().get<float>("dfsph_settings.densityError");
	}
	template<> typename getType<parameters::dfsph_settings,parameters::dfsph_settings::divergenceError>::type& get<parameters::dfsph_settings::divergenceError>(){
		return ParameterManager::instance().get<float>("dfsph_settings.divergenceError");
	}
	template<> typename getType<parameters::dfsph_settings,parameters::dfsph_settings::densitySolverIterations>::type& get<parameters::dfsph_settings::densitySolverIterations>(){
		return ParameterManager::instance().get<int32_t>("dfsph_settings.densitySolverIterations");
	}
	template<> typename getType<parameters::dfsph_settings,parameters::dfsph_settings::divergenceSolverIterations>::type& get<parameters::dfsph_settings::divergenceSolverIterations>(){
		return ParameterManager::instance().get<int32_t>("dfsph_settings.divergenceSolverIterations");
	}
	template<> typename getType<parameters::dfsph_settings,parameters::dfsph_settings::densityEta>::type& get<parameters::dfsph_settings::densityEta>(){
		return ParameterManager::instance().get<float>("dfsph_settings.densityEta");
	}
	template<> typename getType<parameters::dfsph_settings,parameters::dfsph_settings::divergenceEta>::type& get<parameters::dfsph_settings::divergenceEta>(){
		return ParameterManager::instance().get<float>("dfsph_settings.divergenceEta");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::dfsph_settings ident){
		if(ident == parameters::dfsph_settings::densityError) return std::make_pair(std::string("dfsph_settings"), std::string("densityError"));
		if(ident == parameters::dfsph_settings::divergenceError) return std::make_pair(std::string("dfsph_settings"), std::string("divergenceError"));
		if(ident == parameters::dfsph_settings::densitySolverIterations) return std::make_pair(std::string("dfsph_settings"), std::string("densitySolverIterations"));
		if(ident == parameters::dfsph_settings::divergenceSolverIterations) return std::make_pair(std::string("dfsph_settings"), std::string("divergenceSolverIterations"));
		if(ident == parameters::dfsph_settings::densityEta) return std::make_pair(std::string("dfsph_settings"), std::string("densityEta"));
		if(ident == parameters::dfsph_settings::divergenceEta) return std::make_pair(std::string("dfsph_settings"), std::string("divergenceEta"));
	}
	template<> typename getType<parameters::iisph_settings,parameters::iisph_settings::density_error>::type& get<parameters::iisph_settings::density_error>(){
		return ParameterManager::instance().get<float>("iisph_settings.density_error");
	}
	template<> typename getType<parameters::iisph_settings,parameters::iisph_settings::iterations>::type& get<parameters::iisph_settings::iterations>(){
		return ParameterManager::instance().get<int32_t>("iisph_settings.iterations");
	}
	template<> typename getType<parameters::iisph_settings,parameters::iisph_settings::eta>::type& get<parameters::iisph_settings::eta>(){
		return ParameterManager::instance().get<float>("iisph_settings.eta");
	}
	template<> typename getType<parameters::iisph_settings,parameters::iisph_settings::jacobi_omega>::type& get<parameters::iisph_settings::jacobi_omega>(){
		return ParameterManager::instance().get<float>("iisph_settings.jacobi_omega");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::iisph_settings ident){
		if(ident == parameters::iisph_settings::density_error) return std::make_pair(std::string("iisph_settings"), std::string("density_error"));
		if(ident == parameters::iisph_settings::iterations) return std::make_pair(std::string("iisph_settings"), std::string("iterations"));
		if(ident == parameters::iisph_settings::eta) return std::make_pair(std::string("iisph_settings"), std::string("eta"));
		if(ident == parameters::iisph_settings::jacobi_omega) return std::make_pair(std::string("iisph_settings"), std::string("jacobi_omega"));
	}
	template<> typename getType<parameters::inlet_volumes,parameters::inlet_volumes::volume>::type& get<parameters::inlet_volumes::volume>(){
		return ParameterManager::instance().get<std::vector<inletVolume>>("inlet_volumes.volume");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::inlet_volumes ident){
		if(ident == parameters::inlet_volumes::volume) return std::make_pair(std::string("inlet_volumes"), std::string("volume"));
	}
	template<> typename getType<parameters::internal,parameters::internal::neighborhood_kind>::type& get<parameters::internal::neighborhood_kind>(){
		return ParameterManager::instance().get<neighbor_list>("internal.neighborhood_kind");
	}
	template<> typename getType<parameters::internal,parameters::internal::dumpNextframe>::type& get<parameters::internal::dumpNextframe>(){
		return ParameterManager::instance().get<int32_t>("internal.dumpNextframe");
	}
	template<> typename getType<parameters::internal,parameters::internal::dumpForSSSPH>::type& get<parameters::internal::dumpForSSSPH>(){
		return ParameterManager::instance().get<int32_t>("internal.dumpForSSSPH");
	}
	template<> typename getType<parameters::internal,parameters::internal::target>::type& get<parameters::internal::target>(){
		return ParameterManager::instance().get<launch_config>("internal.target");
	}
	template<> typename getType<parameters::internal,parameters::internal::hash_size>::type& get<parameters::internal::hash_size>(){
		return ParameterManager::instance().get<hash_length>("internal.hash_size");
	}
	template<> typename getType<parameters::internal,parameters::internal::cell_ordering>::type& get<parameters::internal::cell_ordering>(){
		return ParameterManager::instance().get<cell_ordering>("internal.cell_ordering");
	}
	template<> typename getType<parameters::internal,parameters::internal::cell_structure>::type& get<parameters::internal::cell_structure>(){
		return ParameterManager::instance().get<cell_structuring>("internal.cell_structure");
	}
	template<> typename getType<parameters::internal,parameters::internal::num_ptcls>::type& get<parameters::internal::num_ptcls>(){
		return ParameterManager::instance().get<int32_t>("internal.num_ptcls");
	}
	template<> typename getType<parameters::internal,parameters::internal::num_ptcls_fluid>::type& get<parameters::internal::num_ptcls_fluid>(){
		return ParameterManager::instance().get<int32_t>("internal.num_ptcls_fluid");
	}
	template<> typename getType<parameters::internal,parameters::internal::folderName>::type& get<parameters::internal::folderName>(){
		return ParameterManager::instance().get<std::string>("internal.folderName");
	}
	template<> typename getType<parameters::internal,parameters::internal::boundaryCounter>::type& get<parameters::internal::boundaryCounter>(){
		return ParameterManager::instance().get<int32_t>("internal.boundaryCounter");
	}
	template<> typename getType<parameters::internal,parameters::internal::boundaryLUTSize>::type& get<parameters::internal::boundaryLUTSize>(){
		return ParameterManager::instance().get<int32_t>("internal.boundaryLUTSize");
	}
	template<> typename getType<parameters::internal,parameters::internal::frame>::type& get<parameters::internal::frame>(){
		return ParameterManager::instance().get<int32_t>("internal.frame");
	}
	template<> typename getType<parameters::internal,parameters::internal::max_velocity>::type& get<parameters::internal::max_velocity>(){
		return ParameterManager::instance().get<float>("internal.max_velocity");
	}
	template<> typename getType<parameters::internal,parameters::internal::min_domain>::type& get<parameters::internal::min_domain>(){
		return ParameterManager::instance().get<float3>("internal.min_domain");
	}
	template<> typename getType<parameters::internal,parameters::internal::max_domain>::type& get<parameters::internal::max_domain>(){
		return ParameterManager::instance().get<float3>("internal.max_domain");
	}
	template<> typename getType<parameters::internal,parameters::internal::min_coord>::type& get<parameters::internal::min_coord>(){
		return ParameterManager::instance().get<float3>("internal.min_coord");
	}
	template<> typename getType<parameters::internal,parameters::internal::max_coord>::type& get<parameters::internal::max_coord>(){
		return ParameterManager::instance().get<float3>("internal.max_coord");
	}
	template<> typename getType<parameters::internal,parameters::internal::cell_size>::type& get<parameters::internal::cell_size>(){
		return ParameterManager::instance().get<float3>("internal.cell_size");
	}
	template<> typename getType<parameters::internal,parameters::internal::gridSize>::type& get<parameters::internal::gridSize>(){
		return ParameterManager::instance().get<int3>("internal.gridSize");
	}
	template<> typename getType<parameters::internal,parameters::internal::ptcl_spacing>::type& get<parameters::internal::ptcl_spacing>(){
		return ParameterManager::instance().get<float>("internal.ptcl_spacing");
	}
	template<> typename getType<parameters::internal,parameters::internal::ptcl_support>::type& get<parameters::internal::ptcl_support>(){
		return ParameterManager::instance().get<float>("internal.ptcl_support");
	}
	template<> typename getType<parameters::internal,parameters::internal::config_file>::type& get<parameters::internal::config_file>(){
		return ParameterManager::instance().get<std::string>("internal.config_file");
	}
	template<> typename getType<parameters::internal,parameters::internal::config_folder>::type& get<parameters::internal::config_folder>(){
		return ParameterManager::instance().get<std::string>("internal.config_folder");
	}
	template<> typename getType<parameters::internal,parameters::internal::working_directory>::type& get<parameters::internal::working_directory>(){
		return ParameterManager::instance().get<std::string>("internal.working_directory");
	}
	template<> typename getType<parameters::internal,parameters::internal::build_directory>::type& get<parameters::internal::build_directory>(){
		return ParameterManager::instance().get<std::string>("internal.build_directory");
	}
	template<> typename getType<parameters::internal,parameters::internal::source_directory>::type& get<parameters::internal::source_directory>(){
		return ParameterManager::instance().get<std::string>("internal.source_directory");
	}
	template<> typename getType<parameters::internal,parameters::internal::binary_directory>::type& get<parameters::internal::binary_directory>(){
		return ParameterManager::instance().get<std::string>("internal.binary_directory");
	}
	template<> typename getType<parameters::internal,parameters::internal::timestep>::type& get<parameters::internal::timestep>(){
		return ParameterManager::instance().get<float>("internal.timestep");
	}
	template<> typename getType<parameters::internal,parameters::internal::simulationTime>::type& get<parameters::internal::simulationTime>(){
		return ParameterManager::instance().get<float>("internal.simulationTime");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::internal ident){
		if(ident == parameters::internal::neighborhood_kind) return std::make_pair(std::string("internal"), std::string("neighborhood_kind"));
		if(ident == parameters::internal::dumpNextframe) return std::make_pair(std::string("internal"), std::string("dumpNextframe"));
		if(ident == parameters::internal::dumpForSSSPH) return std::make_pair(std::string("internal"), std::string("dumpForSSSPH"));
		if(ident == parameters::internal::target) return std::make_pair(std::string("internal"), std::string("target"));
		if(ident == parameters::internal::hash_size) return std::make_pair(std::string("internal"), std::string("hash_size"));
		if(ident == parameters::internal::cell_ordering) return std::make_pair(std::string("internal"), std::string("cell_ordering"));
		if(ident == parameters::internal::cell_structure) return std::make_pair(std::string("internal"), std::string("cell_structure"));
		if(ident == parameters::internal::num_ptcls) return std::make_pair(std::string("internal"), std::string("num_ptcls"));
		if(ident == parameters::internal::num_ptcls_fluid) return std::make_pair(std::string("internal"), std::string("num_ptcls_fluid"));
		if(ident == parameters::internal::folderName) return std::make_pair(std::string("internal"), std::string("folderName"));
		if(ident == parameters::internal::boundaryCounter) return std::make_pair(std::string("internal"), std::string("boundaryCounter"));
		if(ident == parameters::internal::boundaryLUTSize) return std::make_pair(std::string("internal"), std::string("boundaryLUTSize"));
		if(ident == parameters::internal::frame) return std::make_pair(std::string("internal"), std::string("frame"));
		if(ident == parameters::internal::max_velocity) return std::make_pair(std::string("internal"), std::string("max_velocity"));
		if(ident == parameters::internal::min_domain) return std::make_pair(std::string("internal"), std::string("min_domain"));
		if(ident == parameters::internal::max_domain) return std::make_pair(std::string("internal"), std::string("max_domain"));
		if(ident == parameters::internal::min_coord) return std::make_pair(std::string("internal"), std::string("min_coord"));
		if(ident == parameters::internal::max_coord) return std::make_pair(std::string("internal"), std::string("max_coord"));
		if(ident == parameters::internal::cell_size) return std::make_pair(std::string("internal"), std::string("cell_size"));
		if(ident == parameters::internal::gridSize) return std::make_pair(std::string("internal"), std::string("gridSize"));
		if(ident == parameters::internal::ptcl_spacing) return std::make_pair(std::string("internal"), std::string("ptcl_spacing"));
		if(ident == parameters::internal::ptcl_support) return std::make_pair(std::string("internal"), std::string("ptcl_support"));
		if(ident == parameters::internal::config_file) return std::make_pair(std::string("internal"), std::string("config_file"));
		if(ident == parameters::internal::config_folder) return std::make_pair(std::string("internal"), std::string("config_folder"));
		if(ident == parameters::internal::working_directory) return std::make_pair(std::string("internal"), std::string("working_directory"));
		if(ident == parameters::internal::build_directory) return std::make_pair(std::string("internal"), std::string("build_directory"));
		if(ident == parameters::internal::source_directory) return std::make_pair(std::string("internal"), std::string("source_directory"));
		if(ident == parameters::internal::binary_directory) return std::make_pair(std::string("internal"), std::string("binary_directory"));
		if(ident == parameters::internal::timestep) return std::make_pair(std::string("internal"), std::string("timestep"));
		if(ident == parameters::internal::simulationTime) return std::make_pair(std::string("internal"), std::string("simulationTime"));
	}
	template<> typename getType<parameters::modules,parameters::modules::adaptive>::type& get<parameters::modules::adaptive>(){
		return ParameterManager::instance().get<bool>("modules.adaptive");
	}
	template<> typename getType<parameters::modules,parameters::modules::pressure>::type& get<parameters::modules::pressure>(){
		return ParameterManager::instance().get<std::string>("modules.pressure");
	}
	template<> typename getType<parameters::modules,parameters::modules::volumeBoundary>::type& get<parameters::modules::volumeBoundary>(){
		return ParameterManager::instance().get<bool>("modules.volumeBoundary");
	}
	template<> typename getType<parameters::modules,parameters::modules::xsph>::type& get<parameters::modules::xsph>(){
		return ParameterManager::instance().get<bool>("modules.xsph");
	}
	template<> typename getType<parameters::modules,parameters::modules::drag>::type& get<parameters::modules::drag>(){
		return ParameterManager::instance().get<std::string>("modules.drag");
	}
	template<> typename getType<parameters::modules,parameters::modules::viscosity>::type& get<parameters::modules::viscosity>(){
		return ParameterManager::instance().get<bool>("modules.viscosity");
	}
	template<> typename getType<parameters::modules,parameters::modules::tension>::type& get<parameters::modules::tension>(){
		return ParameterManager::instance().get<std::string>("modules.tension");
	}
	template<> typename getType<parameters::modules,parameters::modules::vorticity>::type& get<parameters::modules::vorticity>(){
		return ParameterManager::instance().get<std::string>("modules.vorticity");
	}
	template<> typename getType<parameters::modules,parameters::modules::movingBoundaries>::type& get<parameters::modules::movingBoundaries>(){
		return ParameterManager::instance().get<bool>("modules.movingBoundaries");
	}
	template<> typename getType<parameters::modules,parameters::modules::debug>::type& get<parameters::modules::debug>(){
		return ParameterManager::instance().get<bool>("modules.debug");
	}
	template<> typename getType<parameters::modules,parameters::modules::density>::type& get<parameters::modules::density>(){
		return ParameterManager::instance().get<std::string>("modules.density");
	}
	template<> typename getType<parameters::modules,parameters::modules::particleCleanUp>::type& get<parameters::modules::particleCleanUp>(){
		return ParameterManager::instance().get<bool>("modules.particleCleanUp");
	}
	template<> typename getType<parameters::modules,parameters::modules::volumeInlets>::type& get<parameters::modules::volumeInlets>(){
		return ParameterManager::instance().get<bool>("modules.volumeInlets");
	}
	template<> typename getType<parameters::modules,parameters::modules::volumeOutlets>::type& get<parameters::modules::volumeOutlets>(){
		return ParameterManager::instance().get<bool>("modules.volumeOutlets");
	}
	template<> typename getType<parameters::modules,parameters::modules::logDump>::type& get<parameters::modules::logDump>(){
		return ParameterManager::instance().get<std::string>("modules.logDump");
	}
	template<> typename getType<parameters::modules,parameters::modules::neighborhood>::type& get<parameters::modules::neighborhood>(){
		return ParameterManager::instance().get<std::string>("modules.neighborhood");
	}
	template<> typename getType<parameters::modules,parameters::modules::neighborSorting>::type& get<parameters::modules::neighborSorting>(){
		return ParameterManager::instance().get<bool>("modules.neighborSorting");
	}
	template<> typename getType<parameters::modules,parameters::modules::rayTracing>::type& get<parameters::modules::rayTracing>(){
		return ParameterManager::instance().get<bool>("modules.rayTracing");
	}
	template<> typename getType<parameters::modules,parameters::modules::anisotropicSurface>::type& get<parameters::modules::anisotropicSurface>(){
		return ParameterManager::instance().get<bool>("modules.anisotropicSurface");
	}
	template<> typename getType<parameters::modules,parameters::modules::renderMode>::type& get<parameters::modules::renderMode>(){
		return ParameterManager::instance().get<int32_t>("modules.renderMode");
	}
	template<> typename getType<parameters::modules,parameters::modules::resorting>::type& get<parameters::modules::resorting>(){
		return ParameterManager::instance().get<std::string>("modules.resorting");
	}
	template<> typename getType<parameters::modules,parameters::modules::hash_width>::type& get<parameters::modules::hash_width>(){
		return ParameterManager::instance().get<std::string>("modules.hash_width");
	}
	template<> typename getType<parameters::modules,parameters::modules::alembic>::type& get<parameters::modules::alembic>(){
		return ParameterManager::instance().get<bool>("modules.alembic");
	}
	template<> typename getType<parameters::modules,parameters::modules::error_checking>::type& get<parameters::modules::error_checking>(){
		return ParameterManager::instance().get<bool>("modules.error_checking");
	}
	template<> typename getType<parameters::modules,parameters::modules::gl_record>::type& get<parameters::modules::gl_record>(){
		return ParameterManager::instance().get<bool>("modules.gl_record");
	}
	template<> typename getType<parameters::modules,parameters::modules::launch_cfg>::type& get<parameters::modules::launch_cfg>(){
		return ParameterManager::instance().get<std::string>("modules.launch_cfg");
	}
	template<> typename getType<parameters::modules,parameters::modules::regex_cfg>::type& get<parameters::modules::regex_cfg>(){
		return ParameterManager::instance().get<bool>("modules.regex_cfg");
	}
	template<> typename getType<parameters::modules,parameters::modules::support>::type& get<parameters::modules::support>(){
		return ParameterManager::instance().get<std::string>("modules.support");
	}
	template<> typename getType<parameters::modules,parameters::modules::surfaceDistance>::type& get<parameters::modules::surfaceDistance>(){
		return ParameterManager::instance().get<bool>("modules.surfaceDistance");
	}
	template<> typename getType<parameters::modules,parameters::modules::surfaceDetection>::type& get<parameters::modules::surfaceDetection>(){
		return ParameterManager::instance().get<bool>("modules.surfaceDetection");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::modules ident){
		if(ident == parameters::modules::adaptive) return std::make_pair(std::string("modules"), std::string("adaptive"));
		if(ident == parameters::modules::pressure) return std::make_pair(std::string("modules"), std::string("pressure"));
		if(ident == parameters::modules::volumeBoundary) return std::make_pair(std::string("modules"), std::string("volumeBoundary"));
		if(ident == parameters::modules::xsph) return std::make_pair(std::string("modules"), std::string("xsph"));
		if(ident == parameters::modules::drag) return std::make_pair(std::string("modules"), std::string("drag"));
		if(ident == parameters::modules::viscosity) return std::make_pair(std::string("modules"), std::string("viscosity"));
		if(ident == parameters::modules::tension) return std::make_pair(std::string("modules"), std::string("tension"));
		if(ident == parameters::modules::vorticity) return std::make_pair(std::string("modules"), std::string("vorticity"));
		if(ident == parameters::modules::movingBoundaries) return std::make_pair(std::string("modules"), std::string("movingBoundaries"));
		if(ident == parameters::modules::debug) return std::make_pair(std::string("modules"), std::string("debug"));
		if(ident == parameters::modules::density) return std::make_pair(std::string("modules"), std::string("density"));
		if(ident == parameters::modules::particleCleanUp) return std::make_pair(std::string("modules"), std::string("particleCleanUp"));
		if(ident == parameters::modules::volumeInlets) return std::make_pair(std::string("modules"), std::string("volumeInlets"));
		if(ident == parameters::modules::volumeOutlets) return std::make_pair(std::string("modules"), std::string("volumeOutlets"));
		if(ident == parameters::modules::logDump) return std::make_pair(std::string("modules"), std::string("logDump"));
		if(ident == parameters::modules::neighborhood) return std::make_pair(std::string("modules"), std::string("neighborhood"));
		if(ident == parameters::modules::neighborSorting) return std::make_pair(std::string("modules"), std::string("neighborSorting"));
		if(ident == parameters::modules::rayTracing) return std::make_pair(std::string("modules"), std::string("rayTracing"));
		if(ident == parameters::modules::anisotropicSurface) return std::make_pair(std::string("modules"), std::string("anisotropicSurface"));
		if(ident == parameters::modules::renderMode) return std::make_pair(std::string("modules"), std::string("renderMode"));
		if(ident == parameters::modules::resorting) return std::make_pair(std::string("modules"), std::string("resorting"));
		if(ident == parameters::modules::hash_width) return std::make_pair(std::string("modules"), std::string("hash_width"));
		if(ident == parameters::modules::alembic) return std::make_pair(std::string("modules"), std::string("alembic"));
		if(ident == parameters::modules::error_checking) return std::make_pair(std::string("modules"), std::string("error_checking"));
		if(ident == parameters::modules::gl_record) return std::make_pair(std::string("modules"), std::string("gl_record"));
		if(ident == parameters::modules::launch_cfg) return std::make_pair(std::string("modules"), std::string("launch_cfg"));
		if(ident == parameters::modules::regex_cfg) return std::make_pair(std::string("modules"), std::string("regex_cfg"));
		if(ident == parameters::modules::support) return std::make_pair(std::string("modules"), std::string("support"));
		if(ident == parameters::modules::surfaceDistance) return std::make_pair(std::string("modules"), std::string("surfaceDistance"));
		if(ident == parameters::modules::surfaceDetection) return std::make_pair(std::string("modules"), std::string("surfaceDetection"));
	}
	template<> typename getType<parameters::moving_plane,parameters::moving_plane::plane>::type& get<parameters::moving_plane::plane>(){
		return ParameterManager::instance().get<std::vector<movingPlane>>("moving_plane.plane");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::moving_plane ident){
		if(ident == parameters::moving_plane::plane) return std::make_pair(std::string("moving_plane"), std::string("plane"));
	}
	template<> typename getType<parameters::outlet_volumes,parameters::outlet_volumes::volumeOutletCounter>::type& get<parameters::outlet_volumes::volumeOutletCounter>(){
		return ParameterManager::instance().get<int32_t>("outlet_volumes.volumeOutletCounter");
	}
	template<> typename getType<parameters::outlet_volumes,parameters::outlet_volumes::volumeOutletTime>::type& get<parameters::outlet_volumes::volumeOutletTime>(){
		return ParameterManager::instance().get<float>("outlet_volumes.volumeOutletTime");
	}
	template<> typename getType<parameters::outlet_volumes,parameters::outlet_volumes::volume>::type& get<parameters::outlet_volumes::volume>(){
		return ParameterManager::instance().get<std::vector<outletVolume>>("outlet_volumes.volume");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::outlet_volumes ident){
		if(ident == parameters::outlet_volumes::volumeOutletCounter) return std::make_pair(std::string("outlet_volumes"), std::string("volumeOutletCounter"));
		if(ident == parameters::outlet_volumes::volumeOutletTime) return std::make_pair(std::string("outlet_volumes"), std::string("volumeOutletTime"));
		if(ident == parameters::outlet_volumes::volume) return std::make_pair(std::string("outlet_volumes"), std::string("volume"));
	}
	template<> typename getType<parameters::particleSets,parameters::particleSets::set>::type& get<parameters::particleSets::set>(){
		return ParameterManager::instance().get<std::vector<std::string>>("particleSets.set");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::particleSets ident){
		if(ident == parameters::particleSets::set) return std::make_pair(std::string("particleSets"), std::string("set"));
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::monaghan_viscosity>::type& get<parameters::particle_settings::monaghan_viscosity>(){
		return ParameterManager::instance().get<float>("particle_settings.monaghan_viscosity");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::boundaryViscosity>::type& get<parameters::particle_settings::boundaryViscosity>(){
		return ParameterManager::instance().get<float>("particle_settings.boundaryViscosity");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::xsph_viscosity>::type& get<parameters::particle_settings::xsph_viscosity>(){
		return ParameterManager::instance().get<float>("particle_settings.xsph_viscosity");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::rigidAdhesion_akinci>::type& get<parameters::particle_settings::rigidAdhesion_akinci>(){
		return ParameterManager::instance().get<float>("particle_settings.rigidAdhesion_akinci");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::boundaryAdhesion_akinci>::type& get<parameters::particle_settings::boundaryAdhesion_akinci>(){
		return ParameterManager::instance().get<float>("particle_settings.boundaryAdhesion_akinci");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::tension_akinci>::type& get<parameters::particle_settings::tension_akinci>(){
		return ParameterManager::instance().get<float>("particle_settings.tension_akinci");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::air_velocity>::type& get<parameters::particle_settings::air_velocity>(){
		return ParameterManager::instance().get<float4>("particle_settings.air_velocity");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::radius>::type& get<parameters::particle_settings::radius>(){
		return ParameterManager::instance().get<float>("particle_settings.radius");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::first_fluid>::type& get<parameters::particle_settings::first_fluid>(){
		return ParameterManager::instance().get<int>("particle_settings.first_fluid");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::max_vel>::type& get<parameters::particle_settings::max_vel>(){
		return ParameterManager::instance().get<float>("particle_settings.max_vel");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::min_vel>::type& get<parameters::particle_settings::min_vel>(){
		return ParameterManager::instance().get<float>("particle_settings.min_vel");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::max_neighbors>::type& get<parameters::particle_settings::max_neighbors>(){
		return ParameterManager::instance().get<int>("particle_settings.max_neighbors");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::max_density>::type& get<parameters::particle_settings::max_density>(){
		return ParameterManager::instance().get<float>("particle_settings.max_density");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::sdf_resolution>::type& get<parameters::particle_settings::sdf_resolution>(){
		return ParameterManager::instance().get<int>("particle_settings.sdf_resolution");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::sdf_epsilon>::type& get<parameters::particle_settings::sdf_epsilon>(){
		return ParameterManager::instance().get<float>("particle_settings.sdf_epsilon");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::sdf_minpoint>::type& get<parameters::particle_settings::sdf_minpoint>(){
		return ParameterManager::instance().get<float4>("particle_settings.sdf_minpoint");
	}
	template<> typename getType<parameters::particle_settings,parameters::particle_settings::rest_density>::type& get<parameters::particle_settings::rest_density>(){
		return ParameterManager::instance().get<float>("particle_settings.rest_density");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::particle_settings ident){
		if(ident == parameters::particle_settings::monaghan_viscosity) return std::make_pair(std::string("particle_settings"), std::string("monaghan_viscosity"));
		if(ident == parameters::particle_settings::boundaryViscosity) return std::make_pair(std::string("particle_settings"), std::string("boundaryViscosity"));
		if(ident == parameters::particle_settings::xsph_viscosity) return std::make_pair(std::string("particle_settings"), std::string("xsph_viscosity"));
		if(ident == parameters::particle_settings::rigidAdhesion_akinci) return std::make_pair(std::string("particle_settings"), std::string("rigidAdhesion_akinci"));
		if(ident == parameters::particle_settings::boundaryAdhesion_akinci) return std::make_pair(std::string("particle_settings"), std::string("boundaryAdhesion_akinci"));
		if(ident == parameters::particle_settings::tension_akinci) return std::make_pair(std::string("particle_settings"), std::string("tension_akinci"));
		if(ident == parameters::particle_settings::air_velocity) return std::make_pair(std::string("particle_settings"), std::string("air_velocity"));
		if(ident == parameters::particle_settings::radius) return std::make_pair(std::string("particle_settings"), std::string("radius"));
		if(ident == parameters::particle_settings::first_fluid) return std::make_pair(std::string("particle_settings"), std::string("first_fluid"));
		if(ident == parameters::particle_settings::max_vel) return std::make_pair(std::string("particle_settings"), std::string("max_vel"));
		if(ident == parameters::particle_settings::min_vel) return std::make_pair(std::string("particle_settings"), std::string("min_vel"));
		if(ident == parameters::particle_settings::max_neighbors) return std::make_pair(std::string("particle_settings"), std::string("max_neighbors"));
		if(ident == parameters::particle_settings::max_density) return std::make_pair(std::string("particle_settings"), std::string("max_density"));
		if(ident == parameters::particle_settings::sdf_resolution) return std::make_pair(std::string("particle_settings"), std::string("sdf_resolution"));
		if(ident == parameters::particle_settings::sdf_epsilon) return std::make_pair(std::string("particle_settings"), std::string("sdf_epsilon"));
		if(ident == parameters::particle_settings::sdf_minpoint) return std::make_pair(std::string("particle_settings"), std::string("sdf_minpoint"));
		if(ident == parameters::particle_settings::rest_density) return std::make_pair(std::string("particle_settings"), std::string("rest_density"));
	}
	template<> typename getType<parameters::particle_volumes,parameters::particle_volumes::volume>::type& get<parameters::particle_volumes::volume>(){
		return ParameterManager::instance().get<std::vector<particleVolume>>("particle_volumes.volume");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::particle_volumes ident){
		if(ident == parameters::particle_volumes::volume) return std::make_pair(std::string("particle_volumes"), std::string("volume"));
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::maxAnisotropicSupport>::type& get<parameters::render_settings::maxAnisotropicSupport>(){
		return ParameterManager::instance().get<float4>("render_settings.maxAnisotropicSupport");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::apertureRadius>::type& get<parameters::render_settings::apertureRadius>(){
		return ParameterManager::instance().get<float>("render_settings.apertureRadius");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::anisotropicLambda>::type& get<parameters::render_settings::anisotropicLambda>(){
		return ParameterManager::instance().get<float>("render_settings.anisotropicLambda");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::anisotropicNepsilon>::type& get<parameters::render_settings::anisotropicNepsilon>(){
		return ParameterManager::instance().get<int32_t>("render_settings.anisotropicNepsilon");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::anisotropicKs>::type& get<parameters::render_settings::anisotropicKs>(){
		return ParameterManager::instance().get<float>("render_settings.anisotropicKs");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::anisotropicKr>::type& get<parameters::render_settings::anisotropicKr>(){
		return ParameterManager::instance().get<float>("render_settings.anisotropicKr");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::anisotropicKn>::type& get<parameters::render_settings::anisotropicKn>(){
		return ParameterManager::instance().get<float>("render_settings.anisotropicKn");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::focalDistance>::type& get<parameters::render_settings::focalDistance>(){
		return ParameterManager::instance().get<float>("render_settings.focalDistance");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxNeighborLimit>::type& get<parameters::render_settings::vrtxNeighborLimit>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxNeighborLimit");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxFluidBias>::type& get<parameters::render_settings::vrtxFluidBias>(){
		return ParameterManager::instance().get<float>("render_settings.vrtxFluidBias");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxRenderDomainMin>::type& get<parameters::render_settings::vrtxRenderDomainMin>(){
		return ParameterManager::instance().get<float3>("render_settings.vrtxRenderDomainMin");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxRenderDomainMax>::type& get<parameters::render_settings::vrtxRenderDomainMax>(){
		return ParameterManager::instance().get<float3>("render_settings.vrtxRenderDomainMax");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxFlipCameraUp>::type& get<parameters::render_settings::vrtxFlipCameraUp>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxFlipCameraUp");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxSurfaceExtraction>::type& get<parameters::render_settings::vrtxSurfaceExtraction>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxSurfaceExtraction");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxRenderMode>::type& get<parameters::render_settings::vrtxRenderMode>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxRenderMode");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxRenderGrid>::type& get<parameters::render_settings::vrtxRenderGrid>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxRenderGrid");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxRenderFluid>::type& get<parameters::render_settings::vrtxRenderFluid>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxRenderFluid");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxRenderSurface>::type& get<parameters::render_settings::vrtxRenderSurface>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxRenderSurface");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxDisplayStats>::type& get<parameters::render_settings::vrtxDisplayStats>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxDisplayStats");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxRenderBVH>::type& get<parameters::render_settings::vrtxRenderBVH>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxRenderBVH");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxBVHMaterial>::type& get<parameters::render_settings::vrtxBVHMaterial>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxBVHMaterial");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxRenderNormals>::type& get<parameters::render_settings::vrtxRenderNormals>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxRenderNormals");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxMaterial>::type& get<parameters::render_settings::vrtxMaterial>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxMaterial");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxDomainEpsilon>::type& get<parameters::render_settings::vrtxDomainEpsilon>(){
		return ParameterManager::instance().get<float>("render_settings.vrtxDomainEpsilon");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxDomainMin>::type& get<parameters::render_settings::vrtxDomainMin>(){
		return ParameterManager::instance().get<float3>("render_settings.vrtxDomainMin");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxDomainMax>::type& get<parameters::render_settings::vrtxDomainMax>(){
		return ParameterManager::instance().get<float3>("render_settings.vrtxDomainMax");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxDebeerScale>::type& get<parameters::render_settings::vrtxDebeerScale>(){
		return ParameterManager::instance().get<float>("render_settings.vrtxDebeerScale");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxDebeer>::type& get<parameters::render_settings::vrtxDebeer>(){
		return ParameterManager::instance().get<float3>("render_settings.vrtxDebeer");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::bvhColor>::type& get<parameters::render_settings::bvhColor>(){
		return ParameterManager::instance().get<float3>("render_settings.bvhColor");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxFluidColor>::type& get<parameters::render_settings::vrtxFluidColor>(){
		return ParameterManager::instance().get<float3>("render_settings.vrtxFluidColor");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxDepth>::type& get<parameters::render_settings::vrtxDepth>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxDepth");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxDepthScale>::type& get<parameters::render_settings::vrtxDepthScale>(){
		return ParameterManager::instance().get<float>("render_settings.vrtxDepthScale");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxWMin>::type& get<parameters::render_settings::vrtxWMin>(){
		return ParameterManager::instance().get<float>("render_settings.vrtxWMin");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxR>::type& get<parameters::render_settings::vrtxR>(){
		return ParameterManager::instance().get<float>("render_settings.vrtxR");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::camera_fov>::type& get<parameters::render_settings::camera_fov>(){
		return ParameterManager::instance().get<float>("render_settings.camera_fov");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxWMax>::type& get<parameters::render_settings::vrtxWMax>(){
		return ParameterManager::instance().get<float>("render_settings.vrtxWMax");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxBounces>::type& get<parameters::render_settings::vrtxBounces>(){
		return ParameterManager::instance().get<int32_t>("render_settings.vrtxBounces");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::auxScale>::type& get<parameters::render_settings::auxScale>(){
		return ParameterManager::instance().get<float>("render_settings.auxScale");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::vrtxIOR>::type& get<parameters::render_settings::vrtxIOR>(){
		return ParameterManager::instance().get<float>("render_settings.vrtxIOR");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::renderSteps>::type& get<parameters::render_settings::renderSteps>(){
		return ParameterManager::instance().get<int32_t>("render_settings.renderSteps");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::internalLimit>::type& get<parameters::render_settings::internalLimit>(){
		return ParameterManager::instance().get<float>("render_settings.internalLimit");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::auxCellCount>::type& get<parameters::render_settings::auxCellCount>(){
		return ParameterManager::instance().get<int32_t>("render_settings.auxCellCount");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::axesRender>::type& get<parameters::render_settings::axesRender>(){
		return ParameterManager::instance().get<int32_t>("render_settings.axesRender");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::boundsRender>::type& get<parameters::render_settings::boundsRender>(){
		return ParameterManager::instance().get<int32_t>("render_settings.boundsRender");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::floorRender>::type& get<parameters::render_settings::floorRender>(){
		return ParameterManager::instance().get<int32_t>("render_settings.floorRender");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::axesScale>::type& get<parameters::render_settings::axesScale>(){
		return ParameterManager::instance().get<float>("render_settings.axesScale");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::render_clamp>::type& get<parameters::render_settings::render_clamp>(){
		return ParameterManager::instance().get<float3>("render_settings.render_clamp");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::camera_position>::type& get<parameters::render_settings::camera_position>(){
		return ParameterManager::instance().get<float3>("render_settings.camera_position");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::camera_angle>::type& get<parameters::render_settings::camera_angle>(){
		return ParameterManager::instance().get<float3>("render_settings.camera_angle");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::camera_resolution>::type& get<parameters::render_settings::camera_resolution>(){
		return ParameterManager::instance().get<float2>("render_settings.camera_resolution");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::camera_fps>::type& get<parameters::render_settings::camera_fps>(){
		return ParameterManager::instance().get<float>("render_settings.camera_fps");
	}
	template<> typename getType<parameters::render_settings,parameters::render_settings::gl_file>::type& get<parameters::render_settings::gl_file>(){
		return ParameterManager::instance().get<std::string>("render_settings.gl_file");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::render_settings ident){
		if(ident == parameters::render_settings::maxAnisotropicSupport) return std::make_pair(std::string("render_settings"), std::string("maxAnisotropicSupport"));
		if(ident == parameters::render_settings::apertureRadius) return std::make_pair(std::string("render_settings"), std::string("apertureRadius"));
		if(ident == parameters::render_settings::anisotropicLambda) return std::make_pair(std::string("render_settings"), std::string("anisotropicLambda"));
		if(ident == parameters::render_settings::anisotropicNepsilon) return std::make_pair(std::string("render_settings"), std::string("anisotropicNepsilon"));
		if(ident == parameters::render_settings::anisotropicKs) return std::make_pair(std::string("render_settings"), std::string("anisotropicKs"));
		if(ident == parameters::render_settings::anisotropicKr) return std::make_pair(std::string("render_settings"), std::string("anisotropicKr"));
		if(ident == parameters::render_settings::anisotropicKn) return std::make_pair(std::string("render_settings"), std::string("anisotropicKn"));
		if(ident == parameters::render_settings::focalDistance) return std::make_pair(std::string("render_settings"), std::string("focalDistance"));
		if(ident == parameters::render_settings::vrtxNeighborLimit) return std::make_pair(std::string("render_settings"), std::string("vrtxNeighborLimit"));
		if(ident == parameters::render_settings::vrtxFluidBias) return std::make_pair(std::string("render_settings"), std::string("vrtxFluidBias"));
		if(ident == parameters::render_settings::vrtxRenderDomainMin) return std::make_pair(std::string("render_settings"), std::string("vrtxRenderDomainMin"));
		if(ident == parameters::render_settings::vrtxRenderDomainMax) return std::make_pair(std::string("render_settings"), std::string("vrtxRenderDomainMax"));
		if(ident == parameters::render_settings::vrtxFlipCameraUp) return std::make_pair(std::string("render_settings"), std::string("vrtxFlipCameraUp"));
		if(ident == parameters::render_settings::vrtxSurfaceExtraction) return std::make_pair(std::string("render_settings"), std::string("vrtxSurfaceExtraction"));
		if(ident == parameters::render_settings::vrtxRenderMode) return std::make_pair(std::string("render_settings"), std::string("vrtxRenderMode"));
		if(ident == parameters::render_settings::vrtxRenderGrid) return std::make_pair(std::string("render_settings"), std::string("vrtxRenderGrid"));
		if(ident == parameters::render_settings::vrtxRenderFluid) return std::make_pair(std::string("render_settings"), std::string("vrtxRenderFluid"));
		if(ident == parameters::render_settings::vrtxRenderSurface) return std::make_pair(std::string("render_settings"), std::string("vrtxRenderSurface"));
		if(ident == parameters::render_settings::vrtxDisplayStats) return std::make_pair(std::string("render_settings"), std::string("vrtxDisplayStats"));
		if(ident == parameters::render_settings::vrtxRenderBVH) return std::make_pair(std::string("render_settings"), std::string("vrtxRenderBVH"));
		if(ident == parameters::render_settings::vrtxBVHMaterial) return std::make_pair(std::string("render_settings"), std::string("vrtxBVHMaterial"));
		if(ident == parameters::render_settings::vrtxRenderNormals) return std::make_pair(std::string("render_settings"), std::string("vrtxRenderNormals"));
		if(ident == parameters::render_settings::vrtxMaterial) return std::make_pair(std::string("render_settings"), std::string("vrtxMaterial"));
		if(ident == parameters::render_settings::vrtxDomainEpsilon) return std::make_pair(std::string("render_settings"), std::string("vrtxDomainEpsilon"));
		if(ident == parameters::render_settings::vrtxDomainMin) return std::make_pair(std::string("render_settings"), std::string("vrtxDomainMin"));
		if(ident == parameters::render_settings::vrtxDomainMax) return std::make_pair(std::string("render_settings"), std::string("vrtxDomainMax"));
		if(ident == parameters::render_settings::vrtxDebeerScale) return std::make_pair(std::string("render_settings"), std::string("vrtxDebeerScale"));
		if(ident == parameters::render_settings::vrtxDebeer) return std::make_pair(std::string("render_settings"), std::string("vrtxDebeer"));
		if(ident == parameters::render_settings::bvhColor) return std::make_pair(std::string("render_settings"), std::string("bvhColor"));
		if(ident == parameters::render_settings::vrtxFluidColor) return std::make_pair(std::string("render_settings"), std::string("vrtxFluidColor"));
		if(ident == parameters::render_settings::vrtxDepth) return std::make_pair(std::string("render_settings"), std::string("vrtxDepth"));
		if(ident == parameters::render_settings::vrtxDepthScale) return std::make_pair(std::string("render_settings"), std::string("vrtxDepthScale"));
		if(ident == parameters::render_settings::vrtxWMin) return std::make_pair(std::string("render_settings"), std::string("vrtxWMin"));
		if(ident == parameters::render_settings::vrtxR) return std::make_pair(std::string("render_settings"), std::string("vrtxR"));
		if(ident == parameters::render_settings::camera_fov) return std::make_pair(std::string("render_settings"), std::string("camera_fov"));
		if(ident == parameters::render_settings::vrtxWMax) return std::make_pair(std::string("render_settings"), std::string("vrtxWMax"));
		if(ident == parameters::render_settings::vrtxBounces) return std::make_pair(std::string("render_settings"), std::string("vrtxBounces"));
		if(ident == parameters::render_settings::auxScale) return std::make_pair(std::string("render_settings"), std::string("auxScale"));
		if(ident == parameters::render_settings::vrtxIOR) return std::make_pair(std::string("render_settings"), std::string("vrtxIOR"));
		if(ident == parameters::render_settings::renderSteps) return std::make_pair(std::string("render_settings"), std::string("renderSteps"));
		if(ident == parameters::render_settings::internalLimit) return std::make_pair(std::string("render_settings"), std::string("internalLimit"));
		if(ident == parameters::render_settings::auxCellCount) return std::make_pair(std::string("render_settings"), std::string("auxCellCount"));
		if(ident == parameters::render_settings::axesRender) return std::make_pair(std::string("render_settings"), std::string("axesRender"));
		if(ident == parameters::render_settings::boundsRender) return std::make_pair(std::string("render_settings"), std::string("boundsRender"));
		if(ident == parameters::render_settings::floorRender) return std::make_pair(std::string("render_settings"), std::string("floorRender"));
		if(ident == parameters::render_settings::axesScale) return std::make_pair(std::string("render_settings"), std::string("axesScale"));
		if(ident == parameters::render_settings::render_clamp) return std::make_pair(std::string("render_settings"), std::string("render_clamp"));
		if(ident == parameters::render_settings::camera_position) return std::make_pair(std::string("render_settings"), std::string("camera_position"));
		if(ident == parameters::render_settings::camera_angle) return std::make_pair(std::string("render_settings"), std::string("camera_angle"));
		if(ident == parameters::render_settings::camera_resolution) return std::make_pair(std::string("render_settings"), std::string("camera_resolution"));
		if(ident == parameters::render_settings::camera_fps) return std::make_pair(std::string("render_settings"), std::string("camera_fps"));
		if(ident == parameters::render_settings::gl_file) return std::make_pair(std::string("render_settings"), std::string("gl_file"));
	}
	template<> typename getType<parameters::resort,parameters::resort::auxCells>::type& get<parameters::resort::auxCells>(){
		return ParameterManager::instance().get<int>("resort.auxCells");
	}
	template<> typename getType<parameters::resort,parameters::resort::auxCollisions>::type& get<parameters::resort::auxCollisions>(){
		return ParameterManager::instance().get<int>("resort.auxCollisions");
	}
	template<> typename getType<parameters::resort,parameters::resort::resort_algorithm>::type& get<parameters::resort::resort_algorithm>(){
		return ParameterManager::instance().get<int>("resort.resort_algorithm");
	}
	template<> typename getType<parameters::resort,parameters::resort::valid_cells>::type& get<parameters::resort::valid_cells>(){
		return ParameterManager::instance().get<int>("resort.valid_cells");
	}
	template<> typename getType<parameters::resort,parameters::resort::zOrderScale>::type& get<parameters::resort::zOrderScale>(){
		return ParameterManager::instance().get<float>("resort.zOrderScale");
	}
	template<> typename getType<parameters::resort,parameters::resort::collision_cells>::type& get<parameters::resort::collision_cells>(){
		return ParameterManager::instance().get<int>("resort.collision_cells");
	}
	template<> typename getType<parameters::resort,parameters::resort::occupiedCells>::type& get<parameters::resort::occupiedCells>(){
		return ParameterManager::instance().get<std::vector<int32_t>>("resort.occupiedCells");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::resort ident){
		if(ident == parameters::resort::auxCells) return std::make_pair(std::string("resort"), std::string("auxCells"));
		if(ident == parameters::resort::auxCollisions) return std::make_pair(std::string("resort"), std::string("auxCollisions"));
		if(ident == parameters::resort::resort_algorithm) return std::make_pair(std::string("resort"), std::string("resort_algorithm"));
		if(ident == parameters::resort::valid_cells) return std::make_pair(std::string("resort"), std::string("valid_cells"));
		if(ident == parameters::resort::zOrderScale) return std::make_pair(std::string("resort"), std::string("zOrderScale"));
		if(ident == parameters::resort::collision_cells) return std::make_pair(std::string("resort"), std::string("collision_cells"));
		if(ident == parameters::resort::occupiedCells) return std::make_pair(std::string("resort"), std::string("occupiedCells"));
	}
	template<> typename getType<parameters::rigid_volumes,parameters::rigid_volumes::mesh_resolution>::type& get<parameters::rigid_volumes::mesh_resolution>(){
		return ParameterManager::instance().get<int>("rigid_volumes.mesh_resolution");
	}
	template<> typename getType<parameters::rigid_volumes,parameters::rigid_volumes::gamma>::type& get<parameters::rigid_volumes::gamma>(){
		return ParameterManager::instance().get<float>("rigid_volumes.gamma");
	}
	template<> typename getType<parameters::rigid_volumes,parameters::rigid_volumes::beta>::type& get<parameters::rigid_volumes::beta>(){
		return ParameterManager::instance().get<float>("rigid_volumes.beta");
	}
	template<> typename getType<parameters::rigid_volumes,parameters::rigid_volumes::volume>::type& get<parameters::rigid_volumes::volume>(){
		return ParameterManager::instance().get<std::vector<rigidVolume>>("rigid_volumes.volume");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::rigid_volumes ident){
		if(ident == parameters::rigid_volumes::mesh_resolution) return std::make_pair(std::string("rigid_volumes"), std::string("mesh_resolution"));
		if(ident == parameters::rigid_volumes::gamma) return std::make_pair(std::string("rigid_volumes"), std::string("gamma"));
		if(ident == parameters::rigid_volumes::beta) return std::make_pair(std::string("rigid_volumes"), std::string("beta"));
		if(ident == parameters::rigid_volumes::volume) return std::make_pair(std::string("rigid_volumes"), std::string("volume"));
	}
	template<> typename getType<parameters::rtxScene,parameters::rtxScene::sphere>::type& get<parameters::rtxScene::sphere>(){
		return ParameterManager::instance().get<std::vector<rtxSphere>>("rtxScene.sphere");
	}
	template<> typename getType<parameters::rtxScene,parameters::rtxScene::box>::type& get<parameters::rtxScene::box>(){
		return ParameterManager::instance().get<std::vector<rtxBox>>("rtxScene.box");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::rtxScene ident){
		if(ident == parameters::rtxScene::sphere) return std::make_pair(std::string("rtxScene"), std::string("sphere"));
		if(ident == parameters::rtxScene::box) return std::make_pair(std::string("rtxScene"), std::string("box"));
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::external_force>::type& get<parameters::simulation_settings::external_force>(){
		return ParameterManager::instance().get<float4>("simulation_settings.external_force");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::timestep_min>::type& get<parameters::simulation_settings::timestep_min>(){
		return ParameterManager::instance().get<float>("simulation_settings.timestep_min");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::timestep_max>::type& get<parameters::simulation_settings::timestep_max>(){
		return ParameterManager::instance().get<float>("simulation_settings.timestep_max");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::boundaryDampening>::type& get<parameters::simulation_settings::boundaryDampening>(){
		return ParameterManager::instance().get<float>("simulation_settings.boundaryDampening");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::LUTOffset>::type& get<parameters::simulation_settings::LUTOffset>(){
		return ParameterManager::instance().get<float>("simulation_settings.LUTOffset");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::boundaryObject>::type& get<parameters::simulation_settings::boundaryObject>(){
		return ParameterManager::instance().get<std::string>("simulation_settings.boundaryObject");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::domainWalls>::type& get<parameters::simulation_settings::domainWalls>(){
		return ParameterManager::instance().get<std::string>("simulation_settings.domainWalls");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::neighborlimit>::type& get<parameters::simulation_settings::neighborlimit>(){
		return ParameterManager::instance().get<int32_t>("simulation_settings.neighborlimit");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::dumpFile>::type& get<parameters::simulation_settings::dumpFile>(){
		return ParameterManager::instance().get<std::string>("simulation_settings.dumpFile");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::maxNumptcls>::type& get<parameters::simulation_settings::maxNumptcls>(){
		return ParameterManager::instance().get<int32_t>("simulation_settings.maxNumptcls");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::hash_entries>::type& get<parameters::simulation_settings::hash_entries>(){
		return ParameterManager::instance().get<uint32_t>("simulation_settings.hash_entries");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::mlm_schemes>::type& get<parameters::simulation_settings::mlm_schemes>(){
		return ParameterManager::instance().get<uint32_t>("simulation_settings.mlm_schemes");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::deviceRegex>::type& get<parameters::simulation_settings::deviceRegex>(){
		return ParameterManager::instance().get<std::string>("simulation_settings.deviceRegex");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::hostRegex>::type& get<parameters::simulation_settings::hostRegex>(){
		return ParameterManager::instance().get<std::string>("simulation_settings.hostRegex");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::debugRegex>::type& get<parameters::simulation_settings::debugRegex>(){
		return ParameterManager::instance().get<std::string>("simulation_settings.debugRegex");
	}
	template<> typename getType<parameters::simulation_settings,parameters::simulation_settings::densitySteps>::type& get<parameters::simulation_settings::densitySteps>(){
		return ParameterManager::instance().get<int32_t>("simulation_settings.densitySteps");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::simulation_settings ident){
		if(ident == parameters::simulation_settings::external_force) return std::make_pair(std::string("simulation_settings"), std::string("external_force"));
		if(ident == parameters::simulation_settings::timestep_min) return std::make_pair(std::string("simulation_settings"), std::string("timestep_min"));
		if(ident == parameters::simulation_settings::timestep_max) return std::make_pair(std::string("simulation_settings"), std::string("timestep_max"));
		if(ident == parameters::simulation_settings::boundaryDampening) return std::make_pair(std::string("simulation_settings"), std::string("boundaryDampening"));
		if(ident == parameters::simulation_settings::LUTOffset) return std::make_pair(std::string("simulation_settings"), std::string("LUTOffset"));
		if(ident == parameters::simulation_settings::boundaryObject) return std::make_pair(std::string("simulation_settings"), std::string("boundaryObject"));
		if(ident == parameters::simulation_settings::domainWalls) return std::make_pair(std::string("simulation_settings"), std::string("domainWalls"));
		if(ident == parameters::simulation_settings::neighborlimit) return std::make_pair(std::string("simulation_settings"), std::string("neighborlimit"));
		if(ident == parameters::simulation_settings::dumpFile) return std::make_pair(std::string("simulation_settings"), std::string("dumpFile"));
		if(ident == parameters::simulation_settings::maxNumptcls) return std::make_pair(std::string("simulation_settings"), std::string("maxNumptcls"));
		if(ident == parameters::simulation_settings::hash_entries) return std::make_pair(std::string("simulation_settings"), std::string("hash_entries"));
		if(ident == parameters::simulation_settings::mlm_schemes) return std::make_pair(std::string("simulation_settings"), std::string("mlm_schemes"));
		if(ident == parameters::simulation_settings::deviceRegex) return std::make_pair(std::string("simulation_settings"), std::string("deviceRegex"));
		if(ident == parameters::simulation_settings::hostRegex) return std::make_pair(std::string("simulation_settings"), std::string("hostRegex"));
		if(ident == parameters::simulation_settings::debugRegex) return std::make_pair(std::string("simulation_settings"), std::string("debugRegex"));
		if(ident == parameters::simulation_settings::densitySteps) return std::make_pair(std::string("simulation_settings"), std::string("densitySteps"));
	}
	template<> typename getType<parameters::support,parameters::support::support_current_iteration>::type& get<parameters::support::support_current_iteration>(){
		return ParameterManager::instance().get<uint32_t>("support.support_current_iteration");
	}
	template<> typename getType<parameters::support,parameters::support::adjusted_particles>::type& get<parameters::support::adjusted_particles>(){
		return ParameterManager::instance().get<int32_t>("support.adjusted_particles");
	}
	template<> typename getType<parameters::support,parameters::support::omega>::type& get<parameters::support::omega>(){
		return ParameterManager::instance().get<float>("support.omega");
	}
	template<> typename getType<parameters::support,parameters::support::target_neighbors>::type& get<parameters::support::target_neighbors>(){
		return ParameterManager::instance().get<int32_t>("support.target_neighbors");
	}
	template<> typename getType<parameters::support,parameters::support::support_leeway>::type& get<parameters::support::support_leeway>(){
		return ParameterManager::instance().get<int32_t>("support.support_leeway");
	}
	template<> typename getType<parameters::support,parameters::support::overhead_size>::type& get<parameters::support::overhead_size>(){
		return ParameterManager::instance().get<int32_t>("support.overhead_size");
	}
	template<> typename getType<parameters::support,parameters::support::error_factor>::type& get<parameters::support::error_factor>(){
		return ParameterManager::instance().get<int32_t>("support.error_factor");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::support ident){
		if(ident == parameters::support::support_current_iteration) return std::make_pair(std::string("support"), std::string("support_current_iteration"));
		if(ident == parameters::support::adjusted_particles) return std::make_pair(std::string("support"), std::string("adjusted_particles"));
		if(ident == parameters::support::omega) return std::make_pair(std::string("support"), std::string("omega"));
		if(ident == parameters::support::target_neighbors) return std::make_pair(std::string("support"), std::string("target_neighbors"));
		if(ident == parameters::support::support_leeway) return std::make_pair(std::string("support"), std::string("support_leeway"));
		if(ident == parameters::support::overhead_size) return std::make_pair(std::string("support"), std::string("overhead_size"));
		if(ident == parameters::support::error_factor) return std::make_pair(std::string("support"), std::string("error_factor"));
	}
	template<> typename getType<parameters::surfaceDistance,parameters::surfaceDistance::surface_levelLimit>::type& get<parameters::surfaceDistance::surface_levelLimit>(){
		return ParameterManager::instance().get<float>("surfaceDistance.surface_levelLimit");
	}
	template<> typename getType<parameters::surfaceDistance,parameters::surfaceDistance::surface_neighborLimit>::type& get<parameters::surfaceDistance::surface_neighborLimit>(){
		return ParameterManager::instance().get<int32_t>("surfaceDistance.surface_neighborLimit");
	}
	template<> typename getType<parameters::surfaceDistance,parameters::surfaceDistance::surface_phiMin>::type& get<parameters::surfaceDistance::surface_phiMin>(){
		return ParameterManager::instance().get<float>("surfaceDistance.surface_phiMin");
	}
	template<> typename getType<parameters::surfaceDistance,parameters::surfaceDistance::surface_phiChange>::type& get<parameters::surfaceDistance::surface_phiChange>(){
		return ParameterManager::instance().get<float>("surfaceDistance.surface_phiChange");
	}
	template<> typename getType<parameters::surfaceDistance,parameters::surfaceDistance::surface_distanceFieldDistances>::type& get<parameters::surfaceDistance::surface_distanceFieldDistances>(){
		return ParameterManager::instance().get<float3>("surfaceDistance.surface_distanceFieldDistances");
	}
	template<> typename getType<parameters::surfaceDistance,parameters::surfaceDistance::surface_iterations>::type& get<parameters::surfaceDistance::surface_iterations>(){
		return ParameterManager::instance().get<int32_t>("surfaceDistance.surface_iterations");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::surfaceDistance ident){
		if(ident == parameters::surfaceDistance::surface_levelLimit) return std::make_pair(std::string("surfaceDistance"), std::string("surface_levelLimit"));
		if(ident == parameters::surfaceDistance::surface_neighborLimit) return std::make_pair(std::string("surfaceDistance"), std::string("surface_neighborLimit"));
		if(ident == parameters::surfaceDistance::surface_phiMin) return std::make_pair(std::string("surfaceDistance"), std::string("surface_phiMin"));
		if(ident == parameters::surfaceDistance::surface_phiChange) return std::make_pair(std::string("surfaceDistance"), std::string("surface_phiChange"));
		if(ident == parameters::surfaceDistance::surface_distanceFieldDistances) return std::make_pair(std::string("surfaceDistance"), std::string("surface_distanceFieldDistances"));
		if(ident == parameters::surfaceDistance::surface_iterations) return std::make_pair(std::string("surfaceDistance"), std::string("surface_iterations"));
	}
	template<> typename getType<parameters::vorticitySettings,parameters::vorticitySettings::intertiaInverse>::type& get<parameters::vorticitySettings::intertiaInverse>(){
		return ParameterManager::instance().get<float>("vorticitySettings.intertiaInverse");
	}
	template<> typename getType<parameters::vorticitySettings,parameters::vorticitySettings::viscosityOmega>::type& get<parameters::vorticitySettings::viscosityOmega>(){
		return ParameterManager::instance().get<float>("vorticitySettings.viscosityOmega");
	}
	template<> typename getType<parameters::vorticitySettings,parameters::vorticitySettings::vorticityCoeff>::type& get<parameters::vorticitySettings::vorticityCoeff>(){
		return ParameterManager::instance().get<float>("vorticitySettings.vorticityCoeff");
	}
	std::pair<std::string, std::string> getIdentifier(parameters::vorticitySettings ident){
		if(ident == parameters::vorticitySettings::intertiaInverse) return std::make_pair(std::string("vorticitySettings"), std::string("intertiaInverse"));
		if(ident == parameters::vorticitySettings::viscosityOmega) return std::make_pair(std::string("vorticitySettings"), std::string("viscosityOmega"));
		if(ident == parameters::vorticitySettings::vorticityCoeff) return std::make_pair(std::string("vorticitySettings"), std::string("vorticityCoeff"));
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::adaptivityScaling>::type& uGet<parameters::adaptive::adaptivityScaling>(){
		return ParameterManager::instance().uGet<float>("adaptive.adaptivityScaling");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::adaptivityThreshold>::type& uGet<parameters::adaptive::adaptivityThreshold>(){
		return ParameterManager::instance().uGet<float>("adaptive.adaptivityThreshold");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::adaptivityGamma>::type& uGet<parameters::adaptive::adaptivityGamma>(){
		return ParameterManager::instance().uGet<float>("adaptive.adaptivityGamma");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::resolution>::type& uGet<parameters::adaptive::resolution>(){
		return ParameterManager::instance().uGet<float>("adaptive.resolution");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::useVolume>::type& uGet<parameters::adaptive::useVolume>(){
		return ParameterManager::instance().uGet<int32_t>("adaptive.useVolume");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::minVolume>::type& uGet<parameters::adaptive::minVolume>(){
		return ParameterManager::instance().uGet<uFloat<SI::volume>>("adaptive.minVolume");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::detailedAdaptiveStatistics>::type& uGet<parameters::adaptive::detailedAdaptiveStatistics>(){
		return ParameterManager::instance().uGet<int32_t>("adaptive.detailedAdaptiveStatistics");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::ratio>::type& uGet<parameters::adaptive::ratio>(){
		return ParameterManager::instance().uGet<float>("adaptive.ratio");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::blendSteps>::type& uGet<parameters::adaptive::blendSteps>(){
		return ParameterManager::instance().uGet<float>("adaptive.blendSteps");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::delay>::type& uGet<parameters::adaptive::delay>(){
		return ParameterManager::instance().uGet<uFloat<SI::s>>("adaptive.delay");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::splitPtcls>::type& uGet<parameters::adaptive::splitPtcls>(){
		return ParameterManager::instance().uGet<std::vector<int32_t>>("adaptive.splitPtcls");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::blendedPtcls>::type& uGet<parameters::adaptive::blendedPtcls>(){
		return ParameterManager::instance().uGet<int32_t>("adaptive.blendedPtcls");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::mergedPtcls>::type& uGet<parameters::adaptive::mergedPtcls>(){
		return ParameterManager::instance().uGet<std::vector<int32_t>>("adaptive.mergedPtcls");
	}
	template<> typename getUType<parameters::adaptive,parameters::adaptive::sharedPtcls>::type& uGet<parameters::adaptive::sharedPtcls>(){
		return ParameterManager::instance().uGet<std::vector<int32_t>>("adaptive.sharedPtcls");
	}
	template<> typename getUType<parameters::alembic,parameters::alembic::file_name>::type& uGet<parameters::alembic::file_name>(){
		return ParameterManager::instance().uGet<std::string>("alembic.file_name");
	}
	template<> typename getUType<parameters::alembic,parameters::alembic::fps>::type& uGet<parameters::alembic::fps>(){
		return ParameterManager::instance().uGet<int32_t>("alembic.fps");
	}
	template<> typename getUType<parameters::boundary_volumes,parameters::boundary_volumes::volumeBoundaryCounter>::type& uGet<parameters::boundary_volumes::volumeBoundaryCounter>(){
		return ParameterManager::instance().uGet<int32_t>("boundary_volumes.volumeBoundaryCounter");
	}
	template<> typename getUType<parameters::boundary_volumes,parameters::boundary_volumes::volume>::type& uGet<parameters::boundary_volumes::volume>(){
		return ParameterManager::instance().uGet<std::vector<boundaryVolume>>("boundary_volumes.volume");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::transfer_mode>::type& uGet<parameters::color_map::transfer_mode>(){
		return ParameterManager::instance().uGet<std::string>("color_map.transfer_mode");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::mapping_mode>::type& uGet<parameters::color_map::mapping_mode>(){
		return ParameterManager::instance().uGet<std::string>("color_map.mapping_mode");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::vectorMode>::type& uGet<parameters::color_map::vectorMode>(){
		return ParameterManager::instance().uGet<std::string>("color_map.vectorMode");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::visualizeDirection>::type& uGet<parameters::color_map::visualizeDirection>(){
		return ParameterManager::instance().uGet<int32_t>("color_map.visualizeDirection");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::vectorScale>::type& uGet<parameters::color_map::vectorScale>(){
		return ParameterManager::instance().uGet<float>("color_map.vectorScale");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::vectorScaling>::type& uGet<parameters::color_map::vectorScaling>(){
		return ParameterManager::instance().uGet<int32_t>("color_map.vectorScaling");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::min>::type& uGet<parameters::color_map::min>(){
		return ParameterManager::instance().uGet<float>("color_map.min");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::max>::type& uGet<parameters::color_map::max>(){
		return ParameterManager::instance().uGet<float>("color_map.max");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::transfer_fn>::type& uGet<parameters::color_map::transfer_fn>(){
		return ParameterManager::instance().uGet<int32_t>("color_map.transfer_fn");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::pruneVoxel>::type& uGet<parameters::color_map::pruneVoxel>(){
		return ParameterManager::instance().uGet<int32_t>("color_map.pruneVoxel");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::mapping_fn>::type& uGet<parameters::color_map::mapping_fn>(){
		return ParameterManager::instance().uGet<int32_t>("color_map.mapping_fn");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::autoScaling>::type& uGet<parameters::color_map::autoScaling>(){
		return ParameterManager::instance().uGet<int>("color_map.autoScaling");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::map_flipped>::type& uGet<parameters::color_map::map_flipped>(){
		return ParameterManager::instance().uGet<int>("color_map.map_flipped");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::buffer>::type& uGet<parameters::color_map::buffer>(){
		return ParameterManager::instance().uGet<std::string>("color_map.buffer");
	}
	template<> typename getUType<parameters::color_map,parameters::color_map::map>::type& uGet<parameters::color_map::map>(){
		return ParameterManager::instance().uGet<std::string>("color_map.map");
	}
	template<> typename getUType<parameters::dfsph_settings,parameters::dfsph_settings::densityError>::type& uGet<parameters::dfsph_settings::densityError>(){
		return ParameterManager::instance().uGet<float>("dfsph_settings.densityError");
	}
	template<> typename getUType<parameters::dfsph_settings,parameters::dfsph_settings::divergenceError>::type& uGet<parameters::dfsph_settings::divergenceError>(){
		return ParameterManager::instance().uGet<uFloat<SI::Hz>>("dfsph_settings.divergenceError");
	}
	template<> typename getUType<parameters::dfsph_settings,parameters::dfsph_settings::densitySolverIterations>::type& uGet<parameters::dfsph_settings::densitySolverIterations>(){
		return ParameterManager::instance().uGet<int32_t>("dfsph_settings.densitySolverIterations");
	}
	template<> typename getUType<parameters::dfsph_settings,parameters::dfsph_settings::divergenceSolverIterations>::type& uGet<parameters::dfsph_settings::divergenceSolverIterations>(){
		return ParameterManager::instance().uGet<int32_t>("dfsph_settings.divergenceSolverIterations");
	}
	template<> typename getUType<parameters::dfsph_settings,parameters::dfsph_settings::densityEta>::type& uGet<parameters::dfsph_settings::densityEta>(){
		return ParameterManager::instance().uGet<float>("dfsph_settings.densityEta");
	}
	template<> typename getUType<parameters::dfsph_settings,parameters::dfsph_settings::divergenceEta>::type& uGet<parameters::dfsph_settings::divergenceEta>(){
		return ParameterManager::instance().uGet<uFloat<SI::Hz>>("dfsph_settings.divergenceEta");
	}
	template<> typename getUType<parameters::iisph_settings,parameters::iisph_settings::density_error>::type& uGet<parameters::iisph_settings::density_error>(){
		return ParameterManager::instance().uGet<float>("iisph_settings.density_error");
	}
	template<> typename getUType<parameters::iisph_settings,parameters::iisph_settings::iterations>::type& uGet<parameters::iisph_settings::iterations>(){
		return ParameterManager::instance().uGet<int32_t>("iisph_settings.iterations");
	}
	template<> typename getUType<parameters::iisph_settings,parameters::iisph_settings::eta>::type& uGet<parameters::iisph_settings::eta>(){
		return ParameterManager::instance().uGet<float>("iisph_settings.eta");
	}
	template<> typename getUType<parameters::iisph_settings,parameters::iisph_settings::jacobi_omega>::type& uGet<parameters::iisph_settings::jacobi_omega>(){
		return ParameterManager::instance().uGet<float>("iisph_settings.jacobi_omega");
	}
	template<> typename getUType<parameters::inlet_volumes,parameters::inlet_volumes::volume>::type& uGet<parameters::inlet_volumes::volume>(){
		return ParameterManager::instance().uGet<std::vector<inletVolume>>("inlet_volumes.volume");
	}
	template<> typename getUType<parameters::internal,parameters::internal::neighborhood_kind>::type& uGet<parameters::internal::neighborhood_kind>(){
		return ParameterManager::instance().uGet<neighbor_list>("internal.neighborhood_kind");
	}
	template<> typename getUType<parameters::internal,parameters::internal::dumpNextframe>::type& uGet<parameters::internal::dumpNextframe>(){
		return ParameterManager::instance().uGet<int32_t>("internal.dumpNextframe");
	}
	template<> typename getUType<parameters::internal,parameters::internal::dumpForSSSPH>::type& uGet<parameters::internal::dumpForSSSPH>(){
		return ParameterManager::instance().uGet<int32_t>("internal.dumpForSSSPH");
	}
	template<> typename getUType<parameters::internal,parameters::internal::target>::type& uGet<parameters::internal::target>(){
		return ParameterManager::instance().uGet<launch_config>("internal.target");
	}
	template<> typename getUType<parameters::internal,parameters::internal::hash_size>::type& uGet<parameters::internal::hash_size>(){
		return ParameterManager::instance().uGet<hash_length>("internal.hash_size");
	}
	template<> typename getUType<parameters::internal,parameters::internal::cell_ordering>::type& uGet<parameters::internal::cell_ordering>(){
		return ParameterManager::instance().uGet<cell_ordering>("internal.cell_ordering");
	}
	template<> typename getUType<parameters::internal,parameters::internal::cell_structure>::type& uGet<parameters::internal::cell_structure>(){
		return ParameterManager::instance().uGet<cell_structuring>("internal.cell_structure");
	}
	template<> typename getUType<parameters::internal,parameters::internal::num_ptcls>::type& uGet<parameters::internal::num_ptcls>(){
		return ParameterManager::instance().uGet<int32_t>("internal.num_ptcls");
	}
	template<> typename getUType<parameters::internal,parameters::internal::num_ptcls_fluid>::type& uGet<parameters::internal::num_ptcls_fluid>(){
		return ParameterManager::instance().uGet<int32_t>("internal.num_ptcls_fluid");
	}
	template<> typename getUType<parameters::internal,parameters::internal::folderName>::type& uGet<parameters::internal::folderName>(){
		return ParameterManager::instance().uGet<std::string>("internal.folderName");
	}
	template<> typename getUType<parameters::internal,parameters::internal::boundaryCounter>::type& uGet<parameters::internal::boundaryCounter>(){
		return ParameterManager::instance().uGet<int32_t>("internal.boundaryCounter");
	}
	template<> typename getUType<parameters::internal,parameters::internal::boundaryLUTSize>::type& uGet<parameters::internal::boundaryLUTSize>(){
		return ParameterManager::instance().uGet<int32_t>("internal.boundaryLUTSize");
	}
	template<> typename getUType<parameters::internal,parameters::internal::frame>::type& uGet<parameters::internal::frame>(){
		return ParameterManager::instance().uGet<int32_t>("internal.frame");
	}
	template<> typename getUType<parameters::internal,parameters::internal::max_velocity>::type& uGet<parameters::internal::max_velocity>(){
		return ParameterManager::instance().uGet<uFloat<SI::velocity>>("internal.max_velocity");
	}
	template<> typename getUType<parameters::internal,parameters::internal::min_domain>::type& uGet<parameters::internal::min_domain>(){
		return ParameterManager::instance().uGet<uFloat3<SI::m>>("internal.min_domain");
	}
	template<> typename getUType<parameters::internal,parameters::internal::max_domain>::type& uGet<parameters::internal::max_domain>(){
		return ParameterManager::instance().uGet<uFloat3<SI::m>>("internal.max_domain");
	}
	template<> typename getUType<parameters::internal,parameters::internal::min_coord>::type& uGet<parameters::internal::min_coord>(){
		return ParameterManager::instance().uGet<uFloat3<SI::m>>("internal.min_coord");
	}
	template<> typename getUType<parameters::internal,parameters::internal::max_coord>::type& uGet<parameters::internal::max_coord>(){
		return ParameterManager::instance().uGet<uFloat3<SI::m>>("internal.max_coord");
	}
	template<> typename getUType<parameters::internal,parameters::internal::cell_size>::type& uGet<parameters::internal::cell_size>(){
		return ParameterManager::instance().uGet<uFloat3<SI::m>>("internal.cell_size");
	}
	template<> typename getUType<parameters::internal,parameters::internal::gridSize>::type& uGet<parameters::internal::gridSize>(){
		return ParameterManager::instance().uGet<int3>("internal.gridSize");
	}
	template<> typename getUType<parameters::internal,parameters::internal::ptcl_spacing>::type& uGet<parameters::internal::ptcl_spacing>(){
		return ParameterManager::instance().uGet<uFloat<SI::m>>("internal.ptcl_spacing");
	}
	template<> typename getUType<parameters::internal,parameters::internal::ptcl_support>::type& uGet<parameters::internal::ptcl_support>(){
		return ParameterManager::instance().uGet<uFloat<SI::m>>("internal.ptcl_support");
	}
	template<> typename getUType<parameters::internal,parameters::internal::config_file>::type& uGet<parameters::internal::config_file>(){
		return ParameterManager::instance().uGet<std::string>("internal.config_file");
	}
	template<> typename getUType<parameters::internal,parameters::internal::config_folder>::type& uGet<parameters::internal::config_folder>(){
		return ParameterManager::instance().uGet<std::string>("internal.config_folder");
	}
	template<> typename getUType<parameters::internal,parameters::internal::working_directory>::type& uGet<parameters::internal::working_directory>(){
		return ParameterManager::instance().uGet<std::string>("internal.working_directory");
	}
	template<> typename getUType<parameters::internal,parameters::internal::build_directory>::type& uGet<parameters::internal::build_directory>(){
		return ParameterManager::instance().uGet<std::string>("internal.build_directory");
	}
	template<> typename getUType<parameters::internal,parameters::internal::source_directory>::type& uGet<parameters::internal::source_directory>(){
		return ParameterManager::instance().uGet<std::string>("internal.source_directory");
	}
	template<> typename getUType<parameters::internal,parameters::internal::binary_directory>::type& uGet<parameters::internal::binary_directory>(){
		return ParameterManager::instance().uGet<std::string>("internal.binary_directory");
	}
	template<> typename getUType<parameters::internal,parameters::internal::timestep>::type& uGet<parameters::internal::timestep>(){
		return ParameterManager::instance().uGet<uFloat<SI::s>>("internal.timestep");
	}
	template<> typename getUType<parameters::internal,parameters::internal::simulationTime>::type& uGet<parameters::internal::simulationTime>(){
		return ParameterManager::instance().uGet<uFloat<SI::s>>("internal.simulationTime");
	}
	template<> typename getUType<parameters::modules,parameters::modules::adaptive>::type& uGet<parameters::modules::adaptive>(){
		return ParameterManager::instance().uGet<bool>("modules.adaptive");
	}
	template<> typename getUType<parameters::modules,parameters::modules::pressure>::type& uGet<parameters::modules::pressure>(){
		return ParameterManager::instance().uGet<std::string>("modules.pressure");
	}
	template<> typename getUType<parameters::modules,parameters::modules::volumeBoundary>::type& uGet<parameters::modules::volumeBoundary>(){
		return ParameterManager::instance().uGet<bool>("modules.volumeBoundary");
	}
	template<> typename getUType<parameters::modules,parameters::modules::xsph>::type& uGet<parameters::modules::xsph>(){
		return ParameterManager::instance().uGet<bool>("modules.xsph");
	}
	template<> typename getUType<parameters::modules,parameters::modules::drag>::type& uGet<parameters::modules::drag>(){
		return ParameterManager::instance().uGet<std::string>("modules.drag");
	}
	template<> typename getUType<parameters::modules,parameters::modules::viscosity>::type& uGet<parameters::modules::viscosity>(){
		return ParameterManager::instance().uGet<bool>("modules.viscosity");
	}
	template<> typename getUType<parameters::modules,parameters::modules::tension>::type& uGet<parameters::modules::tension>(){
		return ParameterManager::instance().uGet<std::string>("modules.tension");
	}
	template<> typename getUType<parameters::modules,parameters::modules::vorticity>::type& uGet<parameters::modules::vorticity>(){
		return ParameterManager::instance().uGet<std::string>("modules.vorticity");
	}
	template<> typename getUType<parameters::modules,parameters::modules::movingBoundaries>::type& uGet<parameters::modules::movingBoundaries>(){
		return ParameterManager::instance().uGet<bool>("modules.movingBoundaries");
	}
	template<> typename getUType<parameters::modules,parameters::modules::debug>::type& uGet<parameters::modules::debug>(){
		return ParameterManager::instance().uGet<bool>("modules.debug");
	}
	template<> typename getUType<parameters::modules,parameters::modules::density>::type& uGet<parameters::modules::density>(){
		return ParameterManager::instance().uGet<std::string>("modules.density");
	}
	template<> typename getUType<parameters::modules,parameters::modules::particleCleanUp>::type& uGet<parameters::modules::particleCleanUp>(){
		return ParameterManager::instance().uGet<bool>("modules.particleCleanUp");
	}
	template<> typename getUType<parameters::modules,parameters::modules::volumeInlets>::type& uGet<parameters::modules::volumeInlets>(){
		return ParameterManager::instance().uGet<bool>("modules.volumeInlets");
	}
	template<> typename getUType<parameters::modules,parameters::modules::volumeOutlets>::type& uGet<parameters::modules::volumeOutlets>(){
		return ParameterManager::instance().uGet<bool>("modules.volumeOutlets");
	}
	template<> typename getUType<parameters::modules,parameters::modules::logDump>::type& uGet<parameters::modules::logDump>(){
		return ParameterManager::instance().uGet<std::string>("modules.logDump");
	}
	template<> typename getUType<parameters::modules,parameters::modules::neighborhood>::type& uGet<parameters::modules::neighborhood>(){
		return ParameterManager::instance().uGet<std::string>("modules.neighborhood");
	}
	template<> typename getUType<parameters::modules,parameters::modules::neighborSorting>::type& uGet<parameters::modules::neighborSorting>(){
		return ParameterManager::instance().uGet<bool>("modules.neighborSorting");
	}
	template<> typename getUType<parameters::modules,parameters::modules::rayTracing>::type& uGet<parameters::modules::rayTracing>(){
		return ParameterManager::instance().uGet<bool>("modules.rayTracing");
	}
	template<> typename getUType<parameters::modules,parameters::modules::anisotropicSurface>::type& uGet<parameters::modules::anisotropicSurface>(){
		return ParameterManager::instance().uGet<bool>("modules.anisotropicSurface");
	}
	template<> typename getUType<parameters::modules,parameters::modules::renderMode>::type& uGet<parameters::modules::renderMode>(){
		return ParameterManager::instance().uGet<int32_t>("modules.renderMode");
	}
	template<> typename getUType<parameters::modules,parameters::modules::resorting>::type& uGet<parameters::modules::resorting>(){
		return ParameterManager::instance().uGet<std::string>("modules.resorting");
	}
	template<> typename getUType<parameters::modules,parameters::modules::hash_width>::type& uGet<parameters::modules::hash_width>(){
		return ParameterManager::instance().uGet<std::string>("modules.hash_width");
	}
	template<> typename getUType<parameters::modules,parameters::modules::alembic>::type& uGet<parameters::modules::alembic>(){
		return ParameterManager::instance().uGet<bool>("modules.alembic");
	}
	template<> typename getUType<parameters::modules,parameters::modules::error_checking>::type& uGet<parameters::modules::error_checking>(){
		return ParameterManager::instance().uGet<bool>("modules.error_checking");
	}
	template<> typename getUType<parameters::modules,parameters::modules::gl_record>::type& uGet<parameters::modules::gl_record>(){
		return ParameterManager::instance().uGet<bool>("modules.gl_record");
	}
	template<> typename getUType<parameters::modules,parameters::modules::launch_cfg>::type& uGet<parameters::modules::launch_cfg>(){
		return ParameterManager::instance().uGet<std::string>("modules.launch_cfg");
	}
	template<> typename getUType<parameters::modules,parameters::modules::regex_cfg>::type& uGet<parameters::modules::regex_cfg>(){
		return ParameterManager::instance().uGet<bool>("modules.regex_cfg");
	}
	template<> typename getUType<parameters::modules,parameters::modules::support>::type& uGet<parameters::modules::support>(){
		return ParameterManager::instance().uGet<std::string>("modules.support");
	}
	template<> typename getUType<parameters::modules,parameters::modules::surfaceDistance>::type& uGet<parameters::modules::surfaceDistance>(){
		return ParameterManager::instance().uGet<bool>("modules.surfaceDistance");
	}
	template<> typename getUType<parameters::modules,parameters::modules::surfaceDetection>::type& uGet<parameters::modules::surfaceDetection>(){
		return ParameterManager::instance().uGet<bool>("modules.surfaceDetection");
	}
	template<> typename getUType<parameters::moving_plane,parameters::moving_plane::plane>::type& uGet<parameters::moving_plane::plane>(){
		return ParameterManager::instance().uGet<std::vector<movingPlane>>("moving_plane.plane");
	}
	template<> typename getUType<parameters::outlet_volumes,parameters::outlet_volumes::volumeOutletCounter>::type& uGet<parameters::outlet_volumes::volumeOutletCounter>(){
		return ParameterManager::instance().uGet<int32_t>("outlet_volumes.volumeOutletCounter");
	}
	template<> typename getUType<parameters::outlet_volumes,parameters::outlet_volumes::volumeOutletTime>::type& uGet<parameters::outlet_volumes::volumeOutletTime>(){
		return ParameterManager::instance().uGet<uFloat<SI::s>>("outlet_volumes.volumeOutletTime");
	}
	template<> typename getUType<parameters::outlet_volumes,parameters::outlet_volumes::volume>::type& uGet<parameters::outlet_volumes::volume>(){
		return ParameterManager::instance().uGet<std::vector<outletVolume>>("outlet_volumes.volume");
	}
	template<> typename getUType<parameters::particleSets,parameters::particleSets::set>::type& uGet<parameters::particleSets::set>(){
		return ParameterManager::instance().uGet<std::vector<std::string>>("particleSets.set");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::monaghan_viscosity>::type& uGet<parameters::particle_settings::monaghan_viscosity>(){
		return ParameterManager::instance().uGet<uFloat<SI::velocity>>("particle_settings.monaghan_viscosity");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::boundaryViscosity>::type& uGet<parameters::particle_settings::boundaryViscosity>(){
		return ParameterManager::instance().uGet<uFloat<SI::velocity>>("particle_settings.boundaryViscosity");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::xsph_viscosity>::type& uGet<parameters::particle_settings::xsph_viscosity>(){
		return ParameterManager::instance().uGet<float>("particle_settings.xsph_viscosity");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::rigidAdhesion_akinci>::type& uGet<parameters::particle_settings::rigidAdhesion_akinci>(){
		return ParameterManager::instance().uGet<uFloat<SI::acceleration>>("particle_settings.rigidAdhesion_akinci");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::boundaryAdhesion_akinci>::type& uGet<parameters::particle_settings::boundaryAdhesion_akinci>(){
		return ParameterManager::instance().uGet<uFloat<SI::acceleration>>("particle_settings.boundaryAdhesion_akinci");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::tension_akinci>::type& uGet<parameters::particle_settings::tension_akinci>(){
		return ParameterManager::instance().uGet<uFloat<SI::acceleration>>("particle_settings.tension_akinci");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::air_velocity>::type& uGet<parameters::particle_settings::air_velocity>(){
		return ParameterManager::instance().uGet<uFloat4<SI::velocity>>("particle_settings.air_velocity");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::radius>::type& uGet<parameters::particle_settings::radius>(){
		return ParameterManager::instance().uGet<uFloat<SI::m>>("particle_settings.radius");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::first_fluid>::type& uGet<parameters::particle_settings::first_fluid>(){
		return ParameterManager::instance().uGet<int>("particle_settings.first_fluid");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::max_vel>::type& uGet<parameters::particle_settings::max_vel>(){
		return ParameterManager::instance().uGet<float>("particle_settings.max_vel");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::min_vel>::type& uGet<parameters::particle_settings::min_vel>(){
		return ParameterManager::instance().uGet<float>("particle_settings.min_vel");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::max_neighbors>::type& uGet<parameters::particle_settings::max_neighbors>(){
		return ParameterManager::instance().uGet<int>("particle_settings.max_neighbors");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::max_density>::type& uGet<parameters::particle_settings::max_density>(){
		return ParameterManager::instance().uGet<float>("particle_settings.max_density");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::sdf_resolution>::type& uGet<parameters::particle_settings::sdf_resolution>(){
		return ParameterManager::instance().uGet<int>("particle_settings.sdf_resolution");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::sdf_epsilon>::type& uGet<parameters::particle_settings::sdf_epsilon>(){
		return ParameterManager::instance().uGet<float>("particle_settings.sdf_epsilon");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::sdf_minpoint>::type& uGet<parameters::particle_settings::sdf_minpoint>(){
		return ParameterManager::instance().uGet<float4>("particle_settings.sdf_minpoint");
	}
	template<> typename getUType<parameters::particle_settings,parameters::particle_settings::rest_density>::type& uGet<parameters::particle_settings::rest_density>(){
		return ParameterManager::instance().uGet<uFloat<SI::density>>("particle_settings.rest_density");
	}
	template<> typename getUType<parameters::particle_volumes,parameters::particle_volumes::volume>::type& uGet<parameters::particle_volumes::volume>(){
		return ParameterManager::instance().uGet<std::vector<particleVolume>>("particle_volumes.volume");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::maxAnisotropicSupport>::type& uGet<parameters::render_settings::maxAnisotropicSupport>(){
		return ParameterManager::instance().uGet<float4>("render_settings.maxAnisotropicSupport");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::apertureRadius>::type& uGet<parameters::render_settings::apertureRadius>(){
		return ParameterManager::instance().uGet<float>("render_settings.apertureRadius");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::anisotropicLambda>::type& uGet<parameters::render_settings::anisotropicLambda>(){
		return ParameterManager::instance().uGet<float>("render_settings.anisotropicLambda");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::anisotropicNepsilon>::type& uGet<parameters::render_settings::anisotropicNepsilon>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.anisotropicNepsilon");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::anisotropicKs>::type& uGet<parameters::render_settings::anisotropicKs>(){
		return ParameterManager::instance().uGet<float>("render_settings.anisotropicKs");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::anisotropicKr>::type& uGet<parameters::render_settings::anisotropicKr>(){
		return ParameterManager::instance().uGet<float>("render_settings.anisotropicKr");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::anisotropicKn>::type& uGet<parameters::render_settings::anisotropicKn>(){
		return ParameterManager::instance().uGet<float>("render_settings.anisotropicKn");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::focalDistance>::type& uGet<parameters::render_settings::focalDistance>(){
		return ParameterManager::instance().uGet<float>("render_settings.focalDistance");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxNeighborLimit>::type& uGet<parameters::render_settings::vrtxNeighborLimit>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxNeighborLimit");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxFluidBias>::type& uGet<parameters::render_settings::vrtxFluidBias>(){
		return ParameterManager::instance().uGet<float>("render_settings.vrtxFluidBias");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxRenderDomainMin>::type& uGet<parameters::render_settings::vrtxRenderDomainMin>(){
		return ParameterManager::instance().uGet<float3>("render_settings.vrtxRenderDomainMin");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxRenderDomainMax>::type& uGet<parameters::render_settings::vrtxRenderDomainMax>(){
		return ParameterManager::instance().uGet<float3>("render_settings.vrtxRenderDomainMax");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxFlipCameraUp>::type& uGet<parameters::render_settings::vrtxFlipCameraUp>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxFlipCameraUp");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxSurfaceExtraction>::type& uGet<parameters::render_settings::vrtxSurfaceExtraction>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxSurfaceExtraction");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxRenderMode>::type& uGet<parameters::render_settings::vrtxRenderMode>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxRenderMode");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxRenderGrid>::type& uGet<parameters::render_settings::vrtxRenderGrid>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxRenderGrid");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxRenderFluid>::type& uGet<parameters::render_settings::vrtxRenderFluid>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxRenderFluid");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxRenderSurface>::type& uGet<parameters::render_settings::vrtxRenderSurface>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxRenderSurface");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxDisplayStats>::type& uGet<parameters::render_settings::vrtxDisplayStats>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxDisplayStats");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxRenderBVH>::type& uGet<parameters::render_settings::vrtxRenderBVH>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxRenderBVH");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxBVHMaterial>::type& uGet<parameters::render_settings::vrtxBVHMaterial>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxBVHMaterial");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxRenderNormals>::type& uGet<parameters::render_settings::vrtxRenderNormals>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxRenderNormals");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxMaterial>::type& uGet<parameters::render_settings::vrtxMaterial>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxMaterial");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxDomainEpsilon>::type& uGet<parameters::render_settings::vrtxDomainEpsilon>(){
		return ParameterManager::instance().uGet<float>("render_settings.vrtxDomainEpsilon");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxDomainMin>::type& uGet<parameters::render_settings::vrtxDomainMin>(){
		return ParameterManager::instance().uGet<float3>("render_settings.vrtxDomainMin");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxDomainMax>::type& uGet<parameters::render_settings::vrtxDomainMax>(){
		return ParameterManager::instance().uGet<float3>("render_settings.vrtxDomainMax");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxDebeerScale>::type& uGet<parameters::render_settings::vrtxDebeerScale>(){
		return ParameterManager::instance().uGet<float>("render_settings.vrtxDebeerScale");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxDebeer>::type& uGet<parameters::render_settings::vrtxDebeer>(){
		return ParameterManager::instance().uGet<float3>("render_settings.vrtxDebeer");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::bvhColor>::type& uGet<parameters::render_settings::bvhColor>(){
		return ParameterManager::instance().uGet<float3>("render_settings.bvhColor");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxFluidColor>::type& uGet<parameters::render_settings::vrtxFluidColor>(){
		return ParameterManager::instance().uGet<float3>("render_settings.vrtxFluidColor");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxDepth>::type& uGet<parameters::render_settings::vrtxDepth>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxDepth");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxDepthScale>::type& uGet<parameters::render_settings::vrtxDepthScale>(){
		return ParameterManager::instance().uGet<float>("render_settings.vrtxDepthScale");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxWMin>::type& uGet<parameters::render_settings::vrtxWMin>(){
		return ParameterManager::instance().uGet<float>("render_settings.vrtxWMin");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxR>::type& uGet<parameters::render_settings::vrtxR>(){
		return ParameterManager::instance().uGet<float>("render_settings.vrtxR");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::camera_fov>::type& uGet<parameters::render_settings::camera_fov>(){
		return ParameterManager::instance().uGet<float>("render_settings.camera_fov");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxWMax>::type& uGet<parameters::render_settings::vrtxWMax>(){
		return ParameterManager::instance().uGet<float>("render_settings.vrtxWMax");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxBounces>::type& uGet<parameters::render_settings::vrtxBounces>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.vrtxBounces");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::auxScale>::type& uGet<parameters::render_settings::auxScale>(){
		return ParameterManager::instance().uGet<float>("render_settings.auxScale");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::vrtxIOR>::type& uGet<parameters::render_settings::vrtxIOR>(){
		return ParameterManager::instance().uGet<float>("render_settings.vrtxIOR");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::renderSteps>::type& uGet<parameters::render_settings::renderSteps>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.renderSteps");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::internalLimit>::type& uGet<parameters::render_settings::internalLimit>(){
		return ParameterManager::instance().uGet<float>("render_settings.internalLimit");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::auxCellCount>::type& uGet<parameters::render_settings::auxCellCount>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.auxCellCount");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::axesRender>::type& uGet<parameters::render_settings::axesRender>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.axesRender");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::boundsRender>::type& uGet<parameters::render_settings::boundsRender>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.boundsRender");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::floorRender>::type& uGet<parameters::render_settings::floorRender>(){
		return ParameterManager::instance().uGet<int32_t>("render_settings.floorRender");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::axesScale>::type& uGet<parameters::render_settings::axesScale>(){
		return ParameterManager::instance().uGet<float>("render_settings.axesScale");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::render_clamp>::type& uGet<parameters::render_settings::render_clamp>(){
		return ParameterManager::instance().uGet<float3>("render_settings.render_clamp");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::camera_position>::type& uGet<parameters::render_settings::camera_position>(){
		return ParameterManager::instance().uGet<uFloat3<SI::m>>("render_settings.camera_position");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::camera_angle>::type& uGet<parameters::render_settings::camera_angle>(){
		return ParameterManager::instance().uGet<float3>("render_settings.camera_angle");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::camera_resolution>::type& uGet<parameters::render_settings::camera_resolution>(){
		return ParameterManager::instance().uGet<float2>("render_settings.camera_resolution");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::camera_fps>::type& uGet<parameters::render_settings::camera_fps>(){
		return ParameterManager::instance().uGet<float>("render_settings.camera_fps");
	}
	template<> typename getUType<parameters::render_settings,parameters::render_settings::gl_file>::type& uGet<parameters::render_settings::gl_file>(){
		return ParameterManager::instance().uGet<std::string>("render_settings.gl_file");
	}
	template<> typename getUType<parameters::resort,parameters::resort::auxCells>::type& uGet<parameters::resort::auxCells>(){
		return ParameterManager::instance().uGet<int>("resort.auxCells");
	}
	template<> typename getUType<parameters::resort,parameters::resort::auxCollisions>::type& uGet<parameters::resort::auxCollisions>(){
		return ParameterManager::instance().uGet<int>("resort.auxCollisions");
	}
	template<> typename getUType<parameters::resort,parameters::resort::resort_algorithm>::type& uGet<parameters::resort::resort_algorithm>(){
		return ParameterManager::instance().uGet<int>("resort.resort_algorithm");
	}
	template<> typename getUType<parameters::resort,parameters::resort::valid_cells>::type& uGet<parameters::resort::valid_cells>(){
		return ParameterManager::instance().uGet<int>("resort.valid_cells");
	}
	template<> typename getUType<parameters::resort,parameters::resort::zOrderScale>::type& uGet<parameters::resort::zOrderScale>(){
		return ParameterManager::instance().uGet<float>("resort.zOrderScale");
	}
	template<> typename getUType<parameters::resort,parameters::resort::collision_cells>::type& uGet<parameters::resort::collision_cells>(){
		return ParameterManager::instance().uGet<int>("resort.collision_cells");
	}
	template<> typename getUType<parameters::resort,parameters::resort::occupiedCells>::type& uGet<parameters::resort::occupiedCells>(){
		return ParameterManager::instance().uGet<std::vector<int32_t>>("resort.occupiedCells");
	}
	template<> typename getUType<parameters::rigid_volumes,parameters::rigid_volumes::mesh_resolution>::type& uGet<parameters::rigid_volumes::mesh_resolution>(){
		return ParameterManager::instance().uGet<int>("rigid_volumes.mesh_resolution");
	}
	template<> typename getUType<parameters::rigid_volumes,parameters::rigid_volumes::gamma>::type& uGet<parameters::rigid_volumes::gamma>(){
		return ParameterManager::instance().uGet<float>("rigid_volumes.gamma");
	}
	template<> typename getUType<parameters::rigid_volumes,parameters::rigid_volumes::beta>::type& uGet<parameters::rigid_volumes::beta>(){
		return ParameterManager::instance().uGet<float>("rigid_volumes.beta");
	}
	template<> typename getUType<parameters::rigid_volumes,parameters::rigid_volumes::volume>::type& uGet<parameters::rigid_volumes::volume>(){
		return ParameterManager::instance().uGet<std::vector<rigidVolume>>("rigid_volumes.volume");
	}
	template<> typename getUType<parameters::rtxScene,parameters::rtxScene::sphere>::type& uGet<parameters::rtxScene::sphere>(){
		return ParameterManager::instance().uGet<std::vector<rtxSphere>>("rtxScene.sphere");
	}
	template<> typename getUType<parameters::rtxScene,parameters::rtxScene::box>::type& uGet<parameters::rtxScene::box>(){
		return ParameterManager::instance().uGet<std::vector<rtxBox>>("rtxScene.box");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::external_force>::type& uGet<parameters::simulation_settings::external_force>(){
		return ParameterManager::instance().uGet<uFloat4<SI::acceleration>>("simulation_settings.external_force");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::timestep_min>::type& uGet<parameters::simulation_settings::timestep_min>(){
		return ParameterManager::instance().uGet<uFloat<SI::s>>("simulation_settings.timestep_min");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::timestep_max>::type& uGet<parameters::simulation_settings::timestep_max>(){
		return ParameterManager::instance().uGet<uFloat<SI::s>>("simulation_settings.timestep_max");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::boundaryDampening>::type& uGet<parameters::simulation_settings::boundaryDampening>(){
		return ParameterManager::instance().uGet<float>("simulation_settings.boundaryDampening");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::LUTOffset>::type& uGet<parameters::simulation_settings::LUTOffset>(){
		return ParameterManager::instance().uGet<float>("simulation_settings.LUTOffset");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::boundaryObject>::type& uGet<parameters::simulation_settings::boundaryObject>(){
		return ParameterManager::instance().uGet<std::string>("simulation_settings.boundaryObject");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::domainWalls>::type& uGet<parameters::simulation_settings::domainWalls>(){
		return ParameterManager::instance().uGet<std::string>("simulation_settings.domainWalls");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::neighborlimit>::type& uGet<parameters::simulation_settings::neighborlimit>(){
		return ParameterManager::instance().uGet<int32_t>("simulation_settings.neighborlimit");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::dumpFile>::type& uGet<parameters::simulation_settings::dumpFile>(){
		return ParameterManager::instance().uGet<std::string>("simulation_settings.dumpFile");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::maxNumptcls>::type& uGet<parameters::simulation_settings::maxNumptcls>(){
		return ParameterManager::instance().uGet<int32_t>("simulation_settings.maxNumptcls");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::hash_entries>::type& uGet<parameters::simulation_settings::hash_entries>(){
		return ParameterManager::instance().uGet<uint32_t>("simulation_settings.hash_entries");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::mlm_schemes>::type& uGet<parameters::simulation_settings::mlm_schemes>(){
		return ParameterManager::instance().uGet<uint32_t>("simulation_settings.mlm_schemes");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::deviceRegex>::type& uGet<parameters::simulation_settings::deviceRegex>(){
		return ParameterManager::instance().uGet<std::string>("simulation_settings.deviceRegex");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::hostRegex>::type& uGet<parameters::simulation_settings::hostRegex>(){
		return ParameterManager::instance().uGet<std::string>("simulation_settings.hostRegex");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::debugRegex>::type& uGet<parameters::simulation_settings::debugRegex>(){
		return ParameterManager::instance().uGet<std::string>("simulation_settings.debugRegex");
	}
	template<> typename getUType<parameters::simulation_settings,parameters::simulation_settings::densitySteps>::type& uGet<parameters::simulation_settings::densitySteps>(){
		return ParameterManager::instance().uGet<int32_t>("simulation_settings.densitySteps");
	}
	template<> typename getUType<parameters::support,parameters::support::support_current_iteration>::type& uGet<parameters::support::support_current_iteration>(){
		return ParameterManager::instance().uGet<uint32_t>("support.support_current_iteration");
	}
	template<> typename getUType<parameters::support,parameters::support::adjusted_particles>::type& uGet<parameters::support::adjusted_particles>(){
		return ParameterManager::instance().uGet<int32_t>("support.adjusted_particles");
	}
	template<> typename getUType<parameters::support,parameters::support::omega>::type& uGet<parameters::support::omega>(){
		return ParameterManager::instance().uGet<float>("support.omega");
	}
	template<> typename getUType<parameters::support,parameters::support::target_neighbors>::type& uGet<parameters::support::target_neighbors>(){
		return ParameterManager::instance().uGet<int32_t>("support.target_neighbors");
	}
	template<> typename getUType<parameters::support,parameters::support::support_leeway>::type& uGet<parameters::support::support_leeway>(){
		return ParameterManager::instance().uGet<int32_t>("support.support_leeway");
	}
	template<> typename getUType<parameters::support,parameters::support::overhead_size>::type& uGet<parameters::support::overhead_size>(){
		return ParameterManager::instance().uGet<int32_t>("support.overhead_size");
	}
	template<> typename getUType<parameters::support,parameters::support::error_factor>::type& uGet<parameters::support::error_factor>(){
		return ParameterManager::instance().uGet<int32_t>("support.error_factor");
	}
	template<> typename getUType<parameters::surfaceDistance,parameters::surfaceDistance::surface_levelLimit>::type& uGet<parameters::surfaceDistance::surface_levelLimit>(){
		return ParameterManager::instance().uGet<uFloat<SI::m>>("surfaceDistance.surface_levelLimit");
	}
	template<> typename getUType<parameters::surfaceDistance,parameters::surfaceDistance::surface_neighborLimit>::type& uGet<parameters::surfaceDistance::surface_neighborLimit>(){
		return ParameterManager::instance().uGet<int32_t>("surfaceDistance.surface_neighborLimit");
	}
	template<> typename getUType<parameters::surfaceDistance,parameters::surfaceDistance::surface_phiMin>::type& uGet<parameters::surfaceDistance::surface_phiMin>(){
		return ParameterManager::instance().uGet<uFloat<SI::m>>("surfaceDistance.surface_phiMin");
	}
	template<> typename getUType<parameters::surfaceDistance,parameters::surfaceDistance::surface_phiChange>::type& uGet<parameters::surfaceDistance::surface_phiChange>(){
		return ParameterManager::instance().uGet<float>("surfaceDistance.surface_phiChange");
	}
	template<> typename getUType<parameters::surfaceDistance,parameters::surfaceDistance::surface_distanceFieldDistances>::type& uGet<parameters::surfaceDistance::surface_distanceFieldDistances>(){
		return ParameterManager::instance().uGet<uFloat3<SI::m>>("surfaceDistance.surface_distanceFieldDistances");
	}
	template<> typename getUType<parameters::surfaceDistance,parameters::surfaceDistance::surface_iterations>::type& uGet<parameters::surfaceDistance::surface_iterations>(){
		return ParameterManager::instance().uGet<int32_t>("surfaceDistance.surface_iterations");
	}
	template<> typename getUType<parameters::vorticitySettings,parameters::vorticitySettings::intertiaInverse>::type& uGet<parameters::vorticitySettings::intertiaInverse>(){
		return ParameterManager::instance().uGet<float>("vorticitySettings.intertiaInverse");
	}
	template<> typename getUType<parameters::vorticitySettings,parameters::vorticitySettings::viscosityOmega>::type& uGet<parameters::vorticitySettings::viscosityOmega>(){
		return ParameterManager::instance().uGet<float>("vorticitySettings.viscosityOmega");
	}
	template<> typename getUType<parameters::vorticitySettings,parameters::vorticitySettings::vorticityCoeff>::type& uGet<parameters::vorticitySettings::vorticityCoeff>(){
		return ParameterManager::instance().uGet<float>("vorticitySettings.vorticityCoeff");
	}
