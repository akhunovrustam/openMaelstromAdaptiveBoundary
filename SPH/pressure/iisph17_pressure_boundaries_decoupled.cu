#include <SPH/pressure/iisph17_pressure_boundaries_decoupled.cuh>
#include <utility/include_all.h>


void SPH::IISPH17_PRESSURE_BOUNDARIES_D::divergence_solve(Memory mem){}

neighFunctionType InitAcc(SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);

    arrays.acceleration_inter[i] = arrays.acceleration[i];
    //if (arras.uid[i] == 0) printf("")
    return;
}

neighFunctionType IISPH_predictAdvection2(SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position));

    arrays.velocity[i] = arrays.velocity[i] + arrays.timestep * arrays.acceleration[i];
    arrays._apparentVolume[i] = arrays.volume[i] / arrays.density[i];

}

neighFunctionType IISPH_predictAdvection2_f(SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position));

    arrays.velocity[i] = arrays.velocity[i] + arrays.timestep * arrays.acceleration_inter[i];
    arrays._apparentVolume[i] = arrays.volume[i] / arrays.density_f[i];
    return;
}

neighFunctionType IISPH_predictAdvection2_b(SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position));

    arrays.velocity[i] = arrays.velocity[i] + arrays.timestep * arrays.acceleration_inter[i];
    arrays._apparentVolume[i] = arrays.volume[i] / arrays.density_b[i];
    return;
}



neighFunctionType IISPH_prepareSolver2_f(SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position), (v_adv, velocity), (i_vol, _apparentVolume), (vol, volume));
    // auto s_i = 1.f / arrays.density[i] - 1.f;
    auto s_i = 1.f - vol[i] / i_vol[i];
    auto oldsi = s_i;

    float a_ii = 0.f;
    float4 kernelSum = { 0.f, 0.f, 0.f, 0.f };

    iterateNeighbors(j) {
        if (arrays.particle_type[j] == 0) {
            auto spikyGradient = GPW_ij;

            kernelSum += (i_vol[j] * spikyGradient).val;
            if (arrays.particle_type[j] == 0)
                a_ii += (i_vol[j] * i_vol[j] / (arrays.rest_density * vol[j]) * math::dot3(spikyGradient, spikyGradient)).val;

            if (arrays.particle_type[i] == 0 || (arrays.particle_type[i] != 0 && arrays.particle_type[j] == 0))
                s_i = s_i - arrays.timestep * i_vol[j] * math::dot3(v_adv[i] - (arrays.particle_type[j] == 0 ? v_adv[j] : 0), spikyGradient);
        }
    }

    if (arrays.particle_type[i] == 0)
        a_ii += math::dot3(kernelSum, kernelSum) / (vol[i] * arrays.rest_density);

    arrays._sourceTerm[i] = s_i;

    arrays._Aii[i].val = (-math::square(arrays.timestep) * i_vol[i] * a_ii).val;
    // printf("diagonal El: %.30f\n", arrays._Aii[i].val);
    arrays.pressure.second[i] = 0.f;
}

neighFunctionType IISPH_prepareSolver2_b(SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position), (v_adv, velocity), (i_vol, _apparentVolume), (vol, volume));
    // auto s_i = 1.f / arrays.density[i] - 1.f;
    auto s_i = 1.f - vol[i] / i_vol[i];
    auto oldsi = s_i;

    float a_ii = 0.f;
    float4 kernelSum = { 0.f, 0.f, 0.f, 0.f };

    iterateNeighbors(j) {
        if (arrays.particle_type[j] != 0) {
            auto spikyGradient = GPW_ij;

            kernelSum += (i_vol[j] * spikyGradient).val;
            if (arrays.particle_type[j] == 0)
                a_ii += (i_vol[j] * i_vol[j] / (arrays.rest_density * vol[j]) * math::dot3(spikyGradient, spikyGradient)).val;

            if (arrays.particle_type[i] == 0 || (arrays.particle_type[i] != 0 && arrays.particle_type[j] == 0))
                s_i = s_i - arrays.timestep * i_vol[j] * math::dot3(v_adv[i] - (arrays.particle_type[j] == 0 ? v_adv[j] : 0), spikyGradient);
        }
    }

    if (arrays.particle_type[i] == 0)
        a_ii += math::dot3(kernelSum, kernelSum) / (vol[i] * arrays.rest_density);

    arrays._sourceTerm[i] = s_i;

    arrays._Aii[i].val = (-math::square(arrays.timestep) * i_vol[i] * a_ii).val;
    // printf("diagonal El: %.30f\n", arrays._Aii[i].val);
    arrays.pressure.second[i] = 0.f;
}

neighFunctionType IISPH_jacobiFirst2_f(const SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position), (i_vol, _apparentVolume), (pressure, pressure.second));

    const auto factor = -1.f * i_vol[i] / (arrays.volume[i] * arrays.rest_density);

    float4 kernelSum = { 0.f, 0.f, 0.f, 0.f };
    float4 kernelSum1 = { 0.f, 0.f, 0.f, 0.f };
    if (arrays.particle_type[i] == 0)
    iterateNeighbors(j)
    {
        if (arrays.particle_type[j] == 0)
            kernelSum += (factor * i_vol[j] * (pressure[i] + pressure[j]) * GPW_ij).val;

    }
    if (arrays.particle_type[i] != 0)
        iterateNeighbors(j)
    {
        //TODO perphaps needed to use only fluid neighbors
        if (arrays.particle_type[j] == 0)
            kernelSum += (factor * i_vol[j] * (pressure[i] + pressure[j]) * GPW_ij).val;
    }

    arrays._predictedAcceleration[i].val = kernelSum;

    //========================== PRESSURE SWAP ========================================
    arrays.pressure.first[i] = arrays.pressure.second[i];
}


neighFunctionType IISPH_jacobiFirst2_b(const SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position), (i_vol, _apparentVolume), (pressure, pressure.second));

    const auto factor = -1.f * i_vol[i] / (arrays.volume[i] * arrays.rest_density);

    float4 kernelSum = { 0.f, 0.f, 0.f, 0.f };
    float4 kernelSum1 = { 0.f, 0.f, 0.f, 0.f };
    if (arrays.particle_type[i] == 0)
    iterateNeighbors(j)
    {
        if (arrays.particle_type[j] != 0)
            kernelSum += (factor * i_vol[j] * (pressure[i] + pressure[j]) * GPW_ij).val;

    }
    if (arrays.particle_type[i] != 0)
        iterateNeighbors(j)
    {
        //TODO perphaps needed to use only fluid neighbors
        if (arrays.particle_type[j] == 0)
            kernelSum += (factor * i_vol[j] * (pressure[i] + pressure[j]) * GPW_ij).val;
    }

    arrays._predictedAcceleration[i].val = kernelSum;

    //========================== PRESSURE SWAP ========================================
    arrays.pressure.first[i] = arrays.pressure.second[i];
}


neighFunctionType IISPH_jacobiSecond2_f(const SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position), (acc, _predictedAcceleration));
    alias_arrays((vol, volume), (i_vol, _apparentVolume));

    int fluidind = 0;
    float kernelSum = 0.f;

    if (arrays.particle_type[i] == 0)
    {
        iterateNeighbors(j)
        {
            if (arrays.particle_type[j] == 0)
                kernelSum += (math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[i] - (arrays.particle_type[j] == 0 ? acc[j] : 0), GPW_ij)).val;

        }
    }
    else
    {
        iterateNeighbors(j)
            if (arrays.particle_type[j] == 0)
            {
                fluidind = j;
                kernelSum += (-math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[j], GPW_ij)).val;
            }
    }

    // TODO: set different omega for fluid and rigid particles
    float omega = 0.5f;

    if (arrays.particle_type[i] != 0) omega = omega * vol[i].val / 0.001;

    auto pressure = (math::max(arrays.pressure.first[i] + omega * (arrays._sourceTerm[i].val - kernelSum) / arrays._Aii[i], 0.f)) + 0._Pa;

    auto residual = (kernelSum - arrays._sourceTerm[i].val) * 1._Pa;

    if (math::abs(arrays._Aii[i]) < 1e-20f || pressure != pressure || pressure > 1e16f || pressure < 0.f) {
        pressure = 0._Pa;
        residual = decltype(residual)(0.f);
    }

    // arrays.a_pressure[i] = 120.f;
    arrays.pressure.second[i] = pressure;
    /*if (pressure.val != 0)
    {
        printf("pressure %f nei %d tp %d\n", pressure.val, arrays.neighborListLength[i], arrays.particle_type[i]);
    }*/
    
    //if (pressure.val != 0)
    //{
    //    printf("pressure %f tp %d src %f ker %f dens %f a %.30f\n", pressure.val, arrays.particle_type[i], arrays._sourceTerm[i].val, kernelSum, arrays.density[i].val, arrays._Aii[i].val);
    //    printf("acc fluid %f dens %f type: %d ind %d\n", acc[fluidind].val.x, arrays.density[fluidind].val, arrays.particle_type[fluidind], fluidind);
    //    /*if (arrays.density[fluidind].val == 0.f) printf("dens ZERO\n");
    //    else printf("dens %.30f\n", arrays.density[fluidind].val);*/
    //}
    arrays._volumeError[i] = math::max(residual, 0.f).val * arrays.volume[i].val;
    // if (isRigid(arrays.particle_type[i])) arrays._volumeError[i] = arrays.eta *  arrays.volume[i].val;
}


neighFunctionType IISPH_jacobiSecond2_b(const SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);
    cache_arrays((pos, position), (acc, _predictedAcceleration));
    alias_arrays((vol, volume), (i_vol, _apparentVolume));

    int fluidind = 0;
    float kernelSum = 0.f;

    if (arrays.particle_type[i] == 0)
    {
        iterateNeighbors(j)
        {
            if (arrays.particle_type[j] != 0)
                kernelSum += (math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[i] - (arrays.particle_type[j] == 0 ? acc[j] : 0), GPW_ij)).val;

        }
    }
    else
    {
        iterateNeighbors(j)
            if (arrays.particle_type[j] == 0)
            {
                fluidind = j;
                kernelSum += (-math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[j], GPW_ij)).val;
            }
    }

    // TODO: set different omega for fluid and rigid particles
    float omega = 0.5f;

    if (arrays.particle_type[i] != 0) omega = omega * vol[i].val / 0.001;

    auto pressure = (math::max(arrays.pressure.first[i] + omega * (arrays._sourceTerm[i].val - kernelSum) / arrays._Aii[i], 0.f)) + 0._Pa;

    auto residual = (kernelSum - arrays._sourceTerm[i].val) * 1._Pa;

    if (math::abs(arrays._Aii[i]) < 1e-20f || pressure != pressure || pressure > 1e16f || pressure < 0.f) {
        pressure = 0._Pa;
        residual = decltype(residual)(0.f);
    }

    // arrays.a_pressure[i] = 120.f;
    arrays.pressure.second[i] = pressure;
    /*if (pressure.val != 0)
    {
        printf("pressure %f nei %d tp %d\n", pressure.val, arrays.neighborListLength[i], arrays.particle_type[i]);
    }*/
    
    //if (pressure.val != 0)
    //{
    //    printf("pressure %f tp %d src %f ker %f dens %f a %.30f\n", pressure.val, arrays.particle_type[i], arrays._sourceTerm[i].val, kernelSum, arrays.density[i].val, arrays._Aii[i].val);
    //    printf("acc fluid %f dens %f type: %d ind %d\n", acc[fluidind].val.x, arrays.density[fluidind].val, arrays.particle_type[fluidind], fluidind);
    //    /*if (arrays.density[fluidind].val == 0.f) printf("dens ZERO\n");
    //    else printf("dens %.30f\n", arrays.density[fluidind].val);*/
    //}
    arrays._volumeError[i] = math::max(residual, 0.f).val * arrays.volume[i].val;
    // if (isRigid(arrays.particle_type[i])) arrays._volumeError[i] = arrays.eta *  arrays.volume[i].val;
}

basicFunctionType updateAccelerations2(SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);

    if (arrays.particle_type[i] != 0) return;

    // printf("accccREal %.20f\n", arrays._predictedAcceleration[i].val.x);
    arrays.acceleration[i] = arrays.acceleration_inter[i];
    //printf("accc x %f y %f z %f pres %f\n", arrays._predictedAcceleration[i].val.x, arrays._predictedAcceleration[i].val.y, arrays._predictedAcceleration[i].val.z, arrays.pressure.second[i].val);
      // arrays.pressure.second[i].val, 
      // arrays._predictedAcceleration[i].val.x, 
      // arrays.a_pressure[i].val);
}


basicFunctionType updateInter(SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory arrays) {
    checkedParticleIdx(i);

    if (arrays.particle_type[i] != 0) return;

    arrays.velocity[i] = arrays.velocity[i] - arrays.timestep * arrays.acceleration_inter[i];
    arrays.acceleration_inter[i] += arrays._predictedAcceleration[i];
    //printf("uid %d accel %f %f %f aparvol %f dens %f densf %f densb %f\n", 
    //    i/*arrays.uid[i]*/, arrays.acceleration_inter[i].val.x, arrays.acceleration_inter[i].val.y, 
    //        arrays.acceleration_inter[i].val.z, arrays._apparentVolume[i].val, arrays.density[i].val, arrays.density_f[i].val, arrays.density_b[i].val);
}

neighFunction(Init, InitAcc, "Initialise acc", caches<float4>{});
neighFunction(Predict_f, IISPH_predictAdvection2_f, "IISPH17_PRESSURE_BOUNDARIES_D: predict Advection", caches<float4>{});
neighFunction(Predict_b, IISPH_predictAdvection2_b, "IISPH17_PRESSURE_BOUNDARIES_D: predict Advection", caches<float4>{});
neighFunction(Prepare_f, IISPH_prepareSolver2_f, "IISPH17_PRESSURE_BOUNDARIES_D: prepare Solver", caches<float4, float4, float, float, float>{});
neighFunction(Prepare_b, IISPH_prepareSolver2_b, "IISPH17_PRESSURE_BOUNDARIES_D: prepare Solver", caches<float4, float4, float, float, float>{});
neighFunction(Jacobi1_f, IISPH_jacobiFirst2_f, "IISPH17_PRESSURE_BOUNDARIES_D: jacobi First", caches<float4, float, float, float>{});
neighFunction(Jacobi1_b, IISPH_jacobiFirst2_b, "IISPH17_PRESSURE_BOUNDARIES_D: jacobi First", caches<float4, float, float, float>{});
neighFunction(Jacobi2_f, IISPH_jacobiSecond2_f, "IISPH17_PRESSURE_BOUNDARIES_D: jacobi Second", caches<float4, float4, float>{});
neighFunction(Jacobi2_b, IISPH_jacobiSecond2_b, "IISPH17_PRESSURE_BOUNDARIES_D: jacobi Second", caches<float4, float4, float>{});
basicFunction(UpdateInter, updateInter, "IISPH17_PRESSURE_MIRRORING_D: updaate Acceleration");
basicFunction(Update, updateAccelerations2, "IISPH17_PRESSURE_BOUNDARIES_D: updaate Acceleration");

float err(SPH::IISPH17_PRESSURE_BOUNDARIES_D::Memory mem) {
    float errfluid = 0.f;
    int fnum = 0;
    for (int i = 0; i < mem.num_ptcls; i++)
    {
        if (mem.particle_type[i] == 0)
        {
            fnum++;
            errfluid += mem._volumeError[i].val;
        }
    }
    return errfluid / fnum;
}

void SPH::IISPH17_PRESSURE_BOUNDARIES_D::pressure_solve(Memory mem) {
    //return;
    bool test = true;
    size_t iteration = 0;
    float limit =   0.000001;
    float blimit =  0.000001;
    std::cout << "============================================================= LIMIT " << limit << "\n";
    std::cout << "pres boundaries decoupled " << limit << " \n";
    launch<Init>(mem.num_ptcls, mem); //velocity, _apparentVolume
    float ferror = 0.f;
    float berror = 0.f;
    int totaliter = 0;
    int smalliter = 0;
    do {
        smalliter = 0;
        launch<Predict_b>(mem.num_ptcls, mem); //velocity, _apparentVolume
        launch<Prepare_b>(mem.num_ptcls, mem); //Aii, sourceTerm, swap pressures
        do {
            totaliter++;
            launch<Jacobi1_b>(mem.num_ptcls, mem); //predictedAcceleration, swap pressure
            launch<Jacobi2_b>(mem.num_ptcls, mem); //pressure, volumeError
            smalliter++;
            berror = err(mem);
            //printf("denser %f\n", density_error);
        } while ((smalliter < 1 || berror > blimit) && smalliter < 20);
        launch<Jacobi1_b>(mem.num_ptcls, mem);
        launch<UpdateInter>(mem.num_ptcls, mem);

        smalliter = 0;
        launch<Predict_f>(mem.num_ptcls, mem); //velocity, _apparentVolume
        launch<Prepare_f>(mem.num_ptcls, mem); //Aii, sourceTerm, swap pressures

        do {
            totaliter++;
            launch<Jacobi1_f>(mem.num_ptcls, mem); //predictedAcceleration, swap pressure
            launch<Jacobi2_f>(mem.num_ptcls, mem); //pressure, volumeError
            smalliter++;
            ferror = err(mem);
            //printf("denser %f\n", density_error);
        } while ((smalliter < 1 || ferror > limit) && smalliter < 20);
        launch<Jacobi1_f>(mem.num_ptcls, mem);
        launch<UpdateInter>(mem.num_ptcls, mem);
        
        test = iteration < 1;
        test = test || ferror > limit || berror > blimit;
        test = test && iteration < 10;
        iteration++;
    } while (test);

    std::cout << "totaliter " << " -> " << totaliter << " iter " << " -> " << iteration << " ferr " << ferror << " berr " << berror << std::endl;

    launch<Update>(mem.num_ptcls, mem);

    //get<parameters::iterations>() = (int32_t)iteration;
    // printf("---------------------------------------------------------- rest: %f\n\n", mem.rest_density.val);
}
