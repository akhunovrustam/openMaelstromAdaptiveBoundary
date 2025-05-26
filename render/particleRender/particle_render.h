#pragma once
#include <GLFW/glfw3.h>
#include <render/util/renderer.h>
//#include <render/qGLWidget/oglwidget.h>

/** This renderer displays all particles in the simulation color coded according
 * to the visualization module and a color map. **/
class ParticleRenderer : public Renderer {
public:
  ParticleRenderer();
  virtual void update() override;
  virtual void render(bool pretty) override;
  virtual bool valid() override;

  cudaGraphicsResource *cuda_vbo_resource;


  GLuint m_posAttr;
  GLuint m_colAttr;
  GLuint IBO;
  GLuint VXO;
  GLuint VUV;
};
