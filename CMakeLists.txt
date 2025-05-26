cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
# CMake policy settings
cmake_policy(VERSION 3.9)                                                # required for boost 1.65.1
cmake_policy(SET CMP0071 NEW)                                            # automoc setting policy introduced in CMake 3.10
SET(ARCH 89)
SET(CMAKE_CUDA_ARCHITECTURES 89)

SET(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake")
SET(GVDB_ROOT_DIR "C:/dev/_output")

message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
# CMake settings to build cuda projects properly and to work well with other depdencies, e.g. boost
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR};${CMAKE_SOURCE_DIR}/cmake") # add local CMake files (e.g. for openVDB)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)                         
SET(CMAKE_INCLUDE_CURRENT_DIR ON)    # needed for QT 
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/lib) 
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
    
string(REPLACE "-O2" "-O0" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-O2" "-O0" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

# Project settings
SET(NAME Maelstrom)                                                    # name of the Project, variable for convenience
project(${NAME} LANGUAGES CXX CUDA)

# steps to determine the GPU architecture of the system to only build one architecture
SET(OUTPUTFILE ${CMAKE_CURRENT_BINARY_DIR}/cuda_script) # No suffix required
SET(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_cuda.cu)
if(${USE_CLANG})
    list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
    SET(CUDA_DIR "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0")
    execute_process(COMMAND clang -fuse-ld=lld -fexceptions --cuda-gpu-arch=sm_89 --cuda-path=${CUDA_DIR} -L${CUDA_DIR}\\lib\\x64 -lcudart ${CUDAFILE} -o ${OUTPUTFILE})
    execute_process(COMMAND clang --version)
else()
execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
execute_process(COMMAND nvcc --version)
endif()
execute_process(COMMAND ${OUTPUTFILE} RESULT_VARIABLE CUDA_RETURN_CODE OUTPUT_VARIABLE ARCH)
if(${CUDA_RETURN_CODE} EQUAL 0)
    SET(CUDA_SUCCESS "TRUE")
else() 
    SET(CUDA_SUCCESS "FALSE")
endif() 
if (${CUDA_SUCCESS})
    message(STATUS "CUDA Architecture: ${ARCH}")
else()
    message(WARNING ${ARCH})
endif()
  
#SET(ARCH 75)

add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)
find_package(CUDAToolkit)

# Package finding for all components required should be build as per the readme.md file
FIND_PACKAGE( Boost         REQUIRED COMPONENTS iostreams system thread program_options filesystem)
FIND_PACKAGE( Bullet        REQUIRED )                                            # Rigid body physics
FIND_PACKAGE( Alembic         REQUIRED )                                    # Required for particle based I/O
FIND_PACKAGE( OpenVDB        REQUIRED )                                    # Required for volume based I/O
FIND_PACKAGE( IlmBase         REQUIRED )                                    # Required for openVDB/alembic
FIND_PACKAGE( OpenEXR 2.5.8         REQUIRED )                                  # Required for openVDB/alembic
FIND_PACKAGE( GVDB            REQUIRED )
FIND_PACKAGE( Blosc            REQUIRED )                                    # Required for openVDB/alembic
FIND_PACKAGE( ZLIB            REQUIRED )                                    # Required for openVDB/alembic
#FIND_PACKAGE( TBB            REQUIRED )                                    # Required for openVDB/alembic and parallel CPU code
FIND_PACKAGE( yaml-cpp        REQUIRED PATHS "C:/Users/rustam/vcpkg/installed/x64-windows/include")
FIND_PACKAGE( glfw3            REQUIRED PATHS "C:/Users/rustam/vcpkg/installed/x64-windows/include")
FIND_PACKAGE( glad            REQUIRED PATHS "C:/Users/rustam/vcpkg/installed/x64-windows/include")

# Configure file to store the binary and source path for resolving file paths
configure_file(cmake/config.h.in config/config.h) 
# Set Include paths, due to everything needing almost everything this is done here
SET(INCLUDE_DIRS
    ${ALEMBIC_INCLUDE_DIR} 
    ${ALEMBIC_INCLUDE_DIRS} 
    ${OPENVDB_INCLUDE_DIR}     
    ${OPENEXR_INCLUDE_DIR}     
    ${OPENEXR_INCLUDE_DIR}/OpenEXR
    ${Boost_INCLUDE_DIR}         
    ${ILMBASE_INCLUDE_DIR}     
    ${ILMBASE_INCLUDE_DIR}/OpenEXR
    ${CMAKE_SOURCE_DIR}     
    ${TBB_INCLUDE_DIR} 
    ${ZLIB_INCLUDE_DIR} 
    ${BLOSC_INCLUDE_DIR}  
    ${CMAKE_SOURCE_DIR}         
    ${CMAKE_BINARY_DIR}   
    ${BULLET_INCLUDE_DIRS}
    ${GVDB_INCLUDE_DIR}
    ${YAML_CPP_INCLUDE_DIR}
    ${GLAD_INCLUDE_DIR}
    ${GLFW3_INCLUDE_DIR}
)
INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
 
IF(WIN32)
        SET(CUDA_SEPARABLE_COMPILATION ON)                                         
        SET(CUDA_HOST_COMPILATION_CPP ON)    
        SET(CUDA_ATTACH_VS_BUILD_RULE_TO_FILE ON)     
        INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        LINK_DIRECTORIES(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        LINK_DIRECTORIES(${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
        #CMAKE is broken with respect to CMAKE_CXX_STANDARD and clang-tidy as it generates -std=c++17, which is unknown, instead of /std:c++17
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest /EHa /bigobj /openmp")
        add_compile_definitions(NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
        SET(CUDA_FLAGS_FOR_RELEASE "${CMAKE_CUDA_FLAGS} -O3 -gencode=arch=compute_${ARCH},code=sm_${ARCH} -gencode=arch=compute_${ARCH},code=compute_${ARCH} --extended-lambda --expt-relaxed-constexpr -ftz=true -keep -use_fast_math -src-in-ptx -lineinfo -Xptxas \"-warn-double-usage\"")
        SET(CUDA_FLAGS_FOR_DEBUG "${CMAKE_CUDA_FLAGS} -g -G -gencode=arch=compute_${ARCH},code=sm_${ARCH} -gencode=arch=compute_${ARCH},code=compute_${ARCH} --extended-lambda --expt-relaxed-constexpr -ftz=true -keep -use_fast_math -src-in-ptx -lineinfo -Xptxas \"-warn-double-usage\"")
        SET(CMAKE_CUDA_FLAGS "${CUDA_FLAGS_FOR_RELEASE}")
ELSE(WIN32)  
    SET(CUDA_SEPARABLE_COMPILATION ON)                                         
    SET(CUDA_HOST_COMPILATION_CPP ON)    
    SET(CUDA_ATTACH_VS_BUILD_RULE_TO_FILE ON)     
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")  
    SET(CMAKE_CXX_STANDARD 20)
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    SET(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "")
    SET(CMAKE_CUDA_FLAGS "")
    SET(CMAKE_CUDA_FLAGS_RELEASE "")
    SET(cuda_flags "-O3 -gencode=arch=compute_${ARCH},code=sm_${ARCH} -gencode=arch=compute_${ARCH},code=compute_${ARCH} --extended-lambda --expt-relaxed-constexpr -keep -src-in-ptx -use_fast_math -lineinfo -Xptxas -warn-double-usage -g -DNDEBUG -std=c++17 -Xcompiler -std=c++17 --compiler-options \"-Wall -Wextra\"")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${cuda_flags}")
ENDIF(WIN32)   
   
# Boost and Intels TBB require certain libraries to be available with no path to function properly so set this up
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS} )                            
LINK_DIRECTORIES(${TBB_LIBRARY_DIRS} )
  
# Macro to get all directories within a subdirectory, e.g. for the SPH modules
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child}) 
      LIST(APPEND dirlist ${child})
    ENDIF() 
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()    
  
# Get list of all directories in SPH and metaCode subfolders
SUBDIRLIST(SPH_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/SPH)
SUBDIRLIST(RENDER_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/render)
SUBDIRLIST(META_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/metaCode)
 
# Macro useful to gather all sources for cuda, c++ and qt code and sort them in the correct source group
macro(_GLOB_SOURCES LIBRARY_NAME)
    file(GLOB_RECURSE CUDA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/*.cu)
    file(GLOB_RECURSE CUDA_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/*.cuh)
    file(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/*.cpp)
    file(GLOB_RECURSE CPP_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/*.h ${LIBRARY_NAME}/*.hpp)
    SET(source_list ${CUDA_SOURCES} ${CUDA_HEADERS} ${CPP_SOURCES} ${CPP_HEADERS})
       
    foreach(source IN LISTS source_list)
        get_filename_component(source_path "${source}" PATH) 
        string(REPLACE "/" "\\" source_path_msvc "${source_path}")
        source_group("${source_path_msvc}" FILES "${source}")
    endforeach()  
endmacro(_GLOB_SOURCES) 
   
# Macro used to link all system libraries, e.g. openVDB, boost and TBB but NOT cuda
macro(_SETUP_DEPENDENCIES MODULE_NAME)
IF(NOT WIN32)
    target_link_libraries(${MODULE_NAME}
        Alembic::Alembic
        # blosc
        # ${OpenVDB_LIBRARIES}
        OpenVDB::openvdb
        /usr/local/lib/libblosc.so
        ${OPENEXR_ILMIMF_LIBRARY_RELEASE}
        ${ALEMBIC_ILMBASE_IMATH_LIB} 
        ${ALEMBIC_ILMBASE_ILMTHREAD_LIB}
        ${ALEMBIC_ILMBASE_IEX_LIB}
        ${ALEMBIC_ILMBASE_IEXMATH_LIB}
        ${ALEMBIC_ILMBASE_HALF_LIB}
        ${ILMBASE_HALF_LIBRARY_RELEASE}  
          ${Boost_SYSTEM_LIBRARY_RELEASE}
          ${Boost_THREAD_LIBRARY_RELEASE} 
        ${Boost_LIBRARIES}  
        Boost::filesystem
        Boost::program_options
        ${GVDB_LIB_DIR}/${GVDB_LIB}
        # ${BLOSC_blosc_LIBRARY}
        TBB::tbb
        ${ZLIB_LIBRARY}     
        ${ARGN}
        CUDA::cudart
        LinearMath Bullet3Common BulletDynamics BulletSoftBody
        # ${BULLET_DYNAMICS_LIBRARY}
        # ${BULLET_COLLISION_LIBRARY}
        # ${BULLET_MATH_LIBRARY}
        # ${BULLET_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        yaml-cpp
        stdc++ m
    )  
    ELSE()
    target_link_libraries(${MODULE_NAME}
        #Alembic::Alembic
        OpenVDB::openvdb
        blosc
        ${OPENEXR_ILMIMF_LIBRARY_RELEASE}
        ${ALEMBIC_ILMBASE_IMATH_LIB} 
        ${ALEMBIC_ILMBASE_ILMTHREAD_LIB}
        ${ALEMBIC_ILMBASE_IEX_LIB}
        ${ALEMBIC_ILMBASE_IEXMATH_LIB}
        ${ALEMBIC_ILMBASE_HALF_LIB}
        ${ILMBASE_HALF_LIBRARY_RELEASE}  
          ${Boost_SYSTEM_LIBRARY_RELEASE}
          ${Boost_THREAD_LIBRARY_RELEASE} 
        ${Boost_LIBRARIES}  
        Boost::filesystem
        Boost::program_options
        ${GVDB_LIB_DIR}/${GVDB_LIB}
        # ${BLOSC_blosc_LIBRARY}
        TBB::tbb
        ${ZLIB_LIBRARY}     
        ${ARGN}
        #CUDA::cudart
        LinearMath Bullet3Common BulletDynamics BulletSoftBody
        # ${BULLET_DYNAMICS_LIBRARY}
        # ${BULLET_COLLISION_LIBRARY}
        # ${BULLET_MATH_LIBRARY}
        # ${BULLET_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        yaml-cpp
    )  
    ENDIF(NOT WIN32)
    target_link_libraries(${MODULE_NAME}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES}
        ${CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES}    
    )
    target_include_directories(${MODULE_NAME} PUBLIC ${INCLUDE_DIRECTORIES})
    IF(NOT WIN32)
        target_link_libraries(${MODULE_NAME} stdc++fs) 
    ENDIF(NOT WIN32)  
endmacro(_SETUP_DEPENDENCIES)  
  
message("CUDA HAAAAALF: ${GVDB_LIB_DIR}/${GVDB_LIB}")

function(buildLibrary LIBRARY_NAME TARGETLIST PREFIX )
    _GLOB_SOURCES("${PREFIX}/${LIBRARY_NAME}")
    IF(WIN32)
        add_library( ${LIBRARY_NAME} STATIC ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS} "utility/sdf.h" "utility/sdf2.h") 
    ELSE()
        add_library( ${LIBRARY_NAME} ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS}) 
    ENDIF(WIN32)
    _SETUP_DEPENDENCIES(${LIBRARY_NAME} ${ARGN})
    IF(WIN32)
        IF("${CPP_HEADERS}" MATCHES "dummy\.h")
            add_custom_command(
                OUTPUT ${PREFIX}/${LIBRARY_NAME}/dummy.h
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/dummy.h ${PREFIX}/${LIBRARY_NAME}/dummy.h
                DEPENDS ${CUDA_SOURCES})
        ENDIF()
    ENDIF()
endfunction(buildLibrary) 
function(buildApplication LIBRARY_NAME PREFIX)
    _GLOB_SOURCES("${PREFIX}/${LIBRARY_NAME}")
    add_executable(${LIBRARY_NAME} ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS}  ${CMAKE_SOURCE_DIR}/app.rc "utility/sdf.h" "utility/sdf2.h")
    _SETUP_DEPENDENCIES(${LIBRARY_NAME} ${ARGN})
endfunction(buildApplication) 

# Build all meta code modules 
foreach(MODULE ${META_MODULES})  
    buildApplication(${MODULE} "metaCode")
    set_target_properties(${MODULE} PROPERTIES FOLDER util)
endforeach(MODULE)    

# adds custom target that contains the meta functionality
add_custom_target(config ALL
    COMMAND ${CMAKE_BINARY_DIR}/bin/arrays ${CMAKE_SOURCE_DIR}/utility/identifier/arrays
    COMMAND ${CMAKE_BINARY_DIR}/bin/functions ${CMAKE_SOURCE_DIR}/functions.json ${CMAKE_SOURCE_DIR}/simulation/functions.h ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_BINARY_DIR}/bin/uniforms ${CMAKE_SOURCE_DIR}/utility/identifier/uniform
)  
SET(SPHModules "")
buildLibrary(math DISCARDED_LIBRARIES "${CMAKE_SOURCE_DIR}") 
buildLibrary(imgui DISCARDED_LIBRARIES "${CMAKE_SOURCE_DIR}"  glfw glad::glad) 
buildLibrary(utility DISCARDED_LIBRARIES "${CMAKE_SOURCE_DIR}") 
buildLibrary(tools DISCARDED_LIBRARIES "${CMAKE_SOURCE_DIR}" imgui) 
buildLibrary(IO    DISCARDED_LIBRARIES "${CMAKE_SOURCE_DIR}" utility tools) 
buildLibrary(simulation DISCARDED_LIBRARIES "${CMAKE_SOURCE_DIR}" IO utility tools math) 

foreach(MODULE ${SPH_MODULES})   
    buildLibrary(${MODULE} SPHModules "SPH")
    set_target_properties(${MODULE} PROPERTIES FOLDER SPH)
endforeach(MODULE)    
foreach(MODULE ${RENDER_MODULES})   
    buildLibrary(${MODULE} DISCARDED_LIBRARIES "render")
    set_target_properties(${MODULE} PROPERTIES FOLDER render)
endforeach(MODULE)    

buildApplication(gui "${CMAKE_SOURCE_DIR}" utility simulation tools imgui) 
buildApplication(consoleParticles "${CMAKE_SOURCE_DIR}" utility simulation Bullet3Common util)
 
add_dependencies(config arrays uniforms functions)
add_dependencies(IO config)
 
# Folder setting up for visual studio
set_target_properties (config PROPERTIES FOLDER util)
set_target_properties (utility PROPERTIES FOLDER util)
set_target_properties (tools PROPERTIES FOLDER util)
set_target_properties (math PROPERTIES FOLDER util)  
set_target_properties (imgui PROPERTIES FOLDER UI)  
set_target_properties (consoleParticles PROPERTIES FOLDER UI)
set_target_properties (gui PROPERTIES FOLDER UI)  
   
IF(WIN32)
    include_directories(${BULLET_INCLUDE_DIRS})
ELSE()
    SET(BULLET_LIB_ROOT "/usr/local/lib")
    #MESSAGE(${BULLET_ROOT})
    SET( BULLET_LIBS 
        BulletCollision
        BulletDynamics
        BulletSoftBody
        LinearMath
        Bullet3Common)
        target_link_libraries(utility ${BULLET_LIBS})
    target_link_libraries(gui ${BULLET_LIBS})
    target_link_libraries(consoleParticles ${BULLET_LIBS})
ENDIF()

target_link_libraries(gui simulation ${BULLET_LIBRARIES} Bullet3Common)
target_link_libraries(utility ${BULLET_LIBRARIES} imgui) 
target_link_libraries(IO ${BULLET_LIBRARIES} Alembic) 
 
foreach(MODULE ${SPH_MODULES})  
    target_link_libraries(gui ${MODULE} glfw glad::glad)  
    target_link_libraries(consoleParticles ${MODULE}) 
endforeach(MODULE)     
foreach(MODULE ${RENDER_MODULES})  
    target_link_libraries(gui ${MODULE})
endforeach(MODULE)    

# file(COPY ${GVDB_ROOT_DIR}/bin/gvdb.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/)
file(COPY ${GVDB_BIN_DIR}/cuda_gvdb_copydata.ptx DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/)
file(COPY ${GVDB_BIN_DIR}/cuda_gvdb_module.ptx DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/)
if(WIN32)
file(COPY ${GVDB_ROOT_DIR}/bin/gvdb.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/)
file(COPY ${GVDB_BIN_DIR}/cuda_gvdb_copydata.ptx DESTINATION ${CMAKE_OUTPUT_DIRECTORY}/)
file(COPY ${GVDB_BIN_DIR}/cuda_gvdb_module.ptx DESTINATION ${CMAKE_OUTPUT_DIRECTORY}/)
endif()