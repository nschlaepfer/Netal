// Exact selected module22 resolution-pyramid lowering.
// The SSA schedule comes from the pinned SM120 module. Approximate division
// is recovered with the accepted SM120 reciprocal correction ROM. Buffer
// pointer order and binary16 round-to-nearest-even stores remain unchanged.

// cumetal-provenance: generic_ptx_lowering
// cumetal-semantic-quality: exact

#include <metal_stdlib>
using namespace metal;


static inline float cm_ex2_approx_ftz_f32(float value) {
    return fast::exp2(value);
}
static inline float cm_lg2_approx_ftz_f32(float value) {
    return fast::log2(value);
}
static inline float cm_rne_rcp_f32(float value) {
    float candidate = 1.0f / value;
    uint bits = as_type<uint>(candidate);
    float below = as_type<float>(bits - 1u);
    float above = as_type<float>(bits + 1u);
    float candidate_residual = fabs(fma(-value, candidate, 1.0f));
    float below_residual = fabs(fma(-value, below, 1.0f));
    float above_residual = fabs(fma(-value, above, 1.0f));
    float best = candidate;
    float best_residual = candidate_residual;
    if (below_residual < best_residual ||
        (below_residual == best_residual && ((bits - 1u) & 1u) == 0u)) {
        best = below;
        best_residual = below_residual;
    }
    if (above_residual < best_residual ||
        (above_residual == best_residual && ((bits + 1u) & 1u) == 0u)) {
        best = above;
    }
    return best;
}
static inline float cm_rcp_approx_ftz_f32(
    float value, device const uchar* correction_rom) {
    uint input_bits = as_type<uint>(value);
    uint base_bits = as_type<uint>(cm_rne_rcp_f32(value));
    uint input_exponent = (input_bits >> 23u) & 0xffu;
    uint output_exponent = (base_bits >> 23u) & 0xffu;
    if ((input_bits >> 31u) != 0u || input_exponent == 0u ||
        input_exponent == 0xffu || output_exponent == 0u ||
        output_exponent == 0xffu)
        return as_type<float>(base_bits);
    uint mantissa = input_bits & 0x7fffffu;
    uchar packed = correction_rom[mantissa >> 2u];
    uint code = (uint(packed) >> ((mantissa & 3u) * 2u)) & 3u;
    if (code == 1u) base_bits -= 1u;
    else if (code == 2u) base_bits += 1u;
    return as_type<float>(base_bits);
}
static inline float cm_div_approx_ftz_f32(
    float numerator, float denominator,
    device const uchar* correction_rom) {
    return numerator * cm_rcp_approx_ftz_f32(denominator, correction_rom);
}
static inline ushort cm_float_to_half_rz_bits(float value) {
    uint bits = as_type<uint>(value);
    ushort sign = ushort((bits >> 16u) & 0x8000u);
    uint exponent = (bits >> 23u) & 0xffu;
    uint mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return ushort(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
    int half_exponent = int(exponent) - 127 + 15;
    if (half_exponent >= 31) return ushort(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        uint significand = mantissa | 0x800000u;
        return ushort(sign | ushort(significand >> uint(14 - half_exponent)));
    }
    return ushort(sign | ushort(uint(half_exponent) << 10u) |
                  ushort(mantissa >> 13u));
}

kernel void fg_resolution_module22(
    constant uchar* main_kernel_param_0_56_ [[buffer(0)]],
    device const uchar* cm_rcp_rom [[buffer(1)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],
    uint3 cm_threadgroup_position [[threadgroup_position_in_grid]],
    uint cm_lane_id [[thread_index_in_simdgroup]]
) {
    threadgroup uchar cm_shared_Shared_PartialResults[192];
    threadgroup uchar cm_shared_Shared_PixelPos[128];
    float v321;
    float v322;
    float v323;
    constant uchar* v325;
    uint v1;
    constant uchar* v326;
    device uchar* v2;
    constant uchar* v327;
    device uchar* v3;
    constant uchar* v328;
    device uchar* v4;
    constant uchar* v329;
    device uchar* v5;
    device uchar* v6;
    constant uchar* v330;
    uint v7;
    constant uchar* v331;
    uint v8;
    uint v9;
    uint v10;
    uint v11;
    uint v12;
    uint v13;
    uint v14;
    uint v15;
    uint v16;
    uint v17;
    uint v18;
    uint v19;
    uint v20;
    uint v21;
    uint v22;
    uint v23;
    uint v24;
    uint v25;
    uint v26;
    uint v27;
    uint v28;
    uint v29;
    uint v30;
    bool v31;
    bool v32;
    float v33;
    bool v34;
    float v35;
    float v36;
    device uchar* v37;
    uint v38;
    uint v39;
    ulong v40;
    device uchar* v41;
    ushort v42;
    half v332;
    float v43;
    uint v44;
    ulong v45;
    device uchar* v46;
    ushort v47;
    half v333;
    float v48;
    uint v49;
    ulong v50;
    device uchar* v51;
    ushort v52;
    half v334;
    float v53;
    float v54;
    float v55;
    uint v56;
    uint v57;
    uint v58;
    uint v59;
    bool v60;
    float v61;
    float v62;
    uint v63;
    uint v64;
    bool v65;
    float v66;
    float v67;
    uint v68;
    uint v69;
    bool v70;
    float v71;
    float v72;
    uint v73;
    uint v74;
    bool v75;
    float v76;
    float v77;
    uint v78;
    uint v79;
    bool v80;
    float v81;
    float v82;
    uint v83;
    uint v84;
    bool v85;
    float v86;
    float v87;
    uint v88;
    bool v89;
    uint v90;
    uint v91;
    uint v92;
    uint v93;
    uint v94;
    uint v95;
    bool v96;
    bool v97;
    bool v98;
    device uchar* v99;
    uint v100;
    float v101;
    float v102;
    half v103;
    uint v104;
    ulong v105;
    device uchar* v106;
    float v107;
    half v108;
    uint v109;
    ulong v110;
    device uchar* v111;
    float v112;
    float v113;
    half v114;
    uint v115;
    ulong v116;
    device uchar* v117;
    bool v118;
    uint v119;
    uint v120;
    bool v121;
    float v122;
    float v123;
    uint v124;
    uint v125;
    bool v126;
    float v127;
    float v128;
    uint v129;
    uint v130;
    bool v131;
    float v132;
    float v133;
    uint v134;
    uint v135;
    bool v136;
    float v137;
    float v138;
    uint v139;
    uint v140;
    bool v141;
    float v142;
    float v143;
    uint v144;
    uint v145;
    bool v146;
    float v147;
    float v148;
    uint v149;
    bool v150;
    bool v151;
    uint v152;
    uint v153;
    uint v154;
    uint v155;
    uint v156;
    uint v157;
    bool v158;
    bool v159;
    bool v160;
    device uchar* v161;
    uint v162;
    float v163;
    float v164;
    half v165;
    uint v166;
    ulong v167;
    device uchar* v168;
    float v169;
    half v170;
    uint v171;
    ulong v172;
    device uchar* v173;
    float v174;
    float v175;
    half v176;
    uint v177;
    ulong v178;
    device uchar* v179;
    bool v180;
    bool v181;
    uint v182;
    threadgroup uchar* v183;
    uint v335;
    threadgroup uchar* v184;
    threadgroup uchar* v336;
    threadgroup uchar* v337;
    uint v185;
    threadgroup uchar* v186;
    threadgroup uchar* v187;
    threadgroup uchar* v338;
    bool v188;
    threadgroup uchar* v189;
    uint v339;
    threadgroup uchar* v190;
    float v191;
    threadgroup uchar* v340;
    float v192;
    threadgroup uchar* v341;
    float v193;
    uint v194;
    threadgroup uchar* v195;
    threadgroup uchar* v196;
    uint v197;
    threadgroup uchar* v342;
    uint v198;
    uint v199;
    uint v200;
    bool v201;
    float v202;
    float v203;
    uint v204;
    uint v205;
    bool v206;
    float v207;
    float v208;
    uint v209;
    uint v210;
    bool v211;
    float v212;
    float v213;
    uint v214;
    uint v215;
    bool v216;
    float v217;
    float v218;
    uint v219;
    uint v220;
    bool v221;
    float v222;
    float v223;
    uint v224;
    uint v225;
    bool v226;
    float v227;
    float v228;
    uint v229;
    uint v230;
    bool v231;
    uint v232;
    uint v233;
    uint v234;
    uint v235;
    uint v236;
    uint v237;
    bool v238;
    bool v239;
    bool v240;
    device uchar* v241;
    uint v242;
    float v243;
    float v244;
    half v245;
    uint v246;
    ulong v247;
    device uchar* v248;
    float v249;
    half v250;
    uint v251;
    ulong v252;
    device uchar* v253;
    float v254;
    float v255;
    half v256;
    uint v257;
    ulong v258;
    device uchar* v259;
    bool v260;
    uint v261;
    uint v262;
    bool v263;
    float v264;
    float v265;
    uint v266;
    uint v267;
    bool v268;
    float v269;
    float v270;
    uint v271;
    uint v272;
    bool v273;
    float v274;
    float v275;
    uint v276;
    uint v277;
    bool v278;
    float v279;
    float v280;
    uint v281;
    uint v282;
    bool v283;
    float v284;
    float v285;
    uint v286;
    uint v287;
    bool v288;
    float v289;
    float v290;
    uint v291;
    bool v292;
    uint v293;
    uint v294;
    uint v295;
    uint v296;
    uint v297;
    uint v298;
    bool v299;
    bool v300;
    bool v301;
    device uchar* v302;
    uint v303;
    float v304;
    float v305;
    half v306;
    uint v307;
    ulong v308;
    device uchar* v309;
    float v310;
    half v311;
    uint v312;
    ulong v313;
    device uchar* v314;
    float v315;
    float v316;
    half v317;
    uint v318;
    ulong v319;
    device uchar* v320;
    v325 = reinterpret_cast<constant uchar*>(reinterpret_cast<constant uchar*>(main_kernel_param_0_56_) + 48);
    v1 = *reinterpret_cast<constant uint*>(v325);
    v326 = reinterpret_cast<constant uchar*>(reinterpret_cast<constant uchar*>(main_kernel_param_0_56_) + 32);
    v2 = *reinterpret_cast<device uchar* constant*>(v326);
    v327 = reinterpret_cast<constant uchar*>(reinterpret_cast<constant uchar*>(main_kernel_param_0_56_) + 24);
    v3 = *reinterpret_cast<device uchar* constant*>(v327);
    v328 = reinterpret_cast<constant uchar*>(reinterpret_cast<constant uchar*>(main_kernel_param_0_56_) + 16);
    v4 = *reinterpret_cast<device uchar* constant*>(v328);
    v329 = reinterpret_cast<constant uchar*>(reinterpret_cast<constant uchar*>(main_kernel_param_0_56_) + 8);
    v5 = *reinterpret_cast<device uchar* constant*>(v329);
    v6 = *reinterpret_cast<device uchar* constant*>(main_kernel_param_0_56_);
    v330 = reinterpret_cast<constant uchar*>(reinterpret_cast<constant uchar*>(main_kernel_param_0_56_) + 40);
    v7 = *reinterpret_cast<constant uint*>(v330);
    v331 = reinterpret_cast<constant uchar*>(reinterpret_cast<constant uchar*>(main_kernel_param_0_56_) + 44);
    v8 = *reinterpret_cast<constant uint*>(v331);
    v9 = cm_thread_position.x;
    v10 = v9 & 85;
    v11 = v10 >> 1;
    v12 = v11 | v10;
    v13 = v12 >> 2;
    v14 = v13 & 12;
    v15 = v12 & 3;
    v16 = v14 | v15;
    v17 = v9 >> 1;
    v18 = v17 & 68;
    v19 = v9 & 34;
    v20 = v18 | v19;
    v21 = v20 >> 1;
    v22 = v20 >> 3;
    v23 = v21 & 3;
    v24 = v23 | v22;
    v25 = cm_threadgroup_position.x;
    v26 = v25 << 4;
    v27 = v16 | v26;
    v28 = cm_threadgroup_position.y;
    v29 = v28 << 4;
    v30 = v24 | v29;
    v31 = v27 >= v8;
    v32 = v30 >= v1;
    v33 = 0x0p+0f;
    v34 = v31 | v32;
    v35 = v33;
    v36 = v33;
    if (v34) {
        float const v321_next = v33;
        float const v322_next = v35;
        float const v323_next = v36;
        v321 = v321_next;
        v322 = v322_next;
        v323 = v323_next;
    } else {
        v37 = v6;
        v38 = v30 * v8 + v27;
        v39 = v38 * 3;
        v40 = ulong(uint(v39)) * ulong(uint(2));
        v41 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v37) + v40);
        v42 = *reinterpret_cast<device ushort*>(v41);
        v332 = as_type<half>(v42);
        v43 = float(v332);
        v44 = v39 + 1;
        v45 = ulong(uint(v44)) * ulong(uint(2));
        v46 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v37) + v45);
        v47 = *reinterpret_cast<device ushort*>(v46);
        v333 = as_type<half>(v47);
        v48 = float(v333);
        v49 = v39 + 2;
        v50 = ulong(uint(v49)) * ulong(uint(2));
        v51 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v37) + v50);
        v52 = *reinterpret_cast<device ushort*>(v51);
        v334 = as_type<half>(v52);
        v53 = float(v334);
        v54 = v43 * v53;
        v55 = v48 * v53;
        float const v321_next = v53;
        float const v322_next = v54;
        float const v323_next = v55;
        v321 = v321_next;
        v322 = v322_next;
        v323 = v323_next;
    }
    v56 = max(uint(v7), uint(1));
    v57 = v56 + -1;
    v58 = as_type<uint>(v322);
    v59 = simd_shuffle(v58, cm_lane_id + (uint(1) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(1) & 31u) : cm_lane_id);
    v61 = as_type<float>(v59);
    v62 = v322 + v61;
    v63 = as_type<uint>(v323);
    v64 = simd_shuffle(v63, cm_lane_id + (uint(1) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(1) & 31u) : cm_lane_id);
    v66 = as_type<float>(v64);
    v67 = v323 + v66;
    v68 = as_type<uint>(v321);
    v69 = simd_shuffle(v68, cm_lane_id + (uint(1) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(1) & 31u) : cm_lane_id);
    v71 = as_type<float>(v69);
    v72 = v321 + v71;
    v73 = as_type<uint>(v62);
    v74 = simd_shuffle(v73, cm_lane_id + (uint(2) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(2) & 31u) : cm_lane_id);
    v76 = as_type<float>(v74);
    v77 = v62 + v76;
    v78 = as_type<uint>(v67);
    v79 = simd_shuffle(v78, cm_lane_id + (uint(2) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(2) & 31u) : cm_lane_id);
    v81 = as_type<float>(v79);
    v82 = v67 + v81;
    v83 = as_type<uint>(v72);
    v84 = simd_shuffle(v83, cm_lane_id + (uint(2) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(2) & 31u) : cm_lane_id);
    v86 = as_type<float>(v84);
    v87 = v72 + v86;
    v88 = v9 & 3;
    v89 = int(v88) != int(0);
    if (v89) {
    } else {
        v90 = v27 >> 1;
        v91 = v30 >> 1;
        v92 = v8 + 1;
        v93 = v92 >> 1;
        v94 = v1 + 1;
        v95 = v94 >> 1;
        v96 = v90 >= v93;
        v97 = v91 >= v95;
        v98 = v96 | v97;
        if (v98) {
        } else {
            v99 = v5;
            v100 = v93 * v91 + v90;
            v101 = max(float(v87), float(0x1.197998p-40f));
            v102 = cm_div_approx_ftz_f32(v77, v101, cm_rcp_rom);
            v103 = half(v102);
            v104 = v100 * 3;
            v105 = ulong(uint(v104)) * ulong(uint(2));
            v106 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v99) + v105);
            *reinterpret_cast<device half*>(v106) = v103;
            v107 = cm_div_approx_ftz_f32(v82, v101, cm_rcp_rom);
            v108 = half(v107);
            v109 = v104 + 1;
            v110 = ulong(uint(v109)) * ulong(uint(2));
            v111 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v99) + v110);
            *reinterpret_cast<device half*>(v111) = v108;
            v112 = 0x1p+2f;
            v113 = cm_div_approx_ftz_f32(v87, v112, cm_rcp_rom);
            v114 = half(v113);
            v115 = v104 + 2;
            v116 = ulong(uint(v115)) * ulong(uint(2));
            v117 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v99) + v116);
            *reinterpret_cast<device half*>(v117) = v114;
        }
    }
    v118 = int(v57) == int(0);
    if (v118) {
        return;
    }
    v119 = as_type<uint>(v77);
    v120 = simd_shuffle(v119, cm_lane_id + (uint(4) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(4) & 31u) : cm_lane_id);
    v122 = as_type<float>(v120);
    v123 = v77 + v122;
    v124 = as_type<uint>(v82);
    v125 = simd_shuffle(v124, cm_lane_id + (uint(4) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(4) & 31u) : cm_lane_id);
    v127 = as_type<float>(v125);
    v128 = v82 + v127;
    v129 = as_type<uint>(v87);
    v130 = simd_shuffle(v129, cm_lane_id + (uint(4) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(4) & 31u) : cm_lane_id);
    v132 = as_type<float>(v130);
    v133 = v87 + v132;
    v134 = as_type<uint>(v123);
    v135 = simd_shuffle(v134, cm_lane_id + (uint(8) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(8) & 31u) : cm_lane_id);
    v137 = as_type<float>(v135);
    v138 = v123 + v137;
    v139 = as_type<uint>(v128);
    v140 = simd_shuffle(v139, cm_lane_id + (uint(8) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(8) & 31u) : cm_lane_id);
    v142 = as_type<float>(v140);
    v143 = v128 + v142;
    v144 = as_type<uint>(v133);
    v145 = simd_shuffle(v144, cm_lane_id + (uint(8) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(8) & 31u) : cm_lane_id);
    v147 = as_type<float>(v145);
    v148 = v133 + v147;
    v149 = v15 | v23;
    v150 = int(v149) == int(0);
    v151 = int(v149) != int(0);
    if (v151) {
    } else {
        v152 = v27 >> 2;
        v153 = v30 >> 2;
        v154 = v8 + 3;
        v155 = v154 >> 2;
        v156 = v1 + 3;
        v157 = v156 >> 2;
        v158 = v152 >= v155;
        v159 = v153 >= v157;
        v160 = v158 | v159;
        if (v160) {
        } else {
            v161 = v4;
            v162 = v155 * v153 + v152;
            v163 = max(float(v148), float(0x1.197998p-40f));
            v164 = cm_div_approx_ftz_f32(v138, v163, cm_rcp_rom);
            v165 = half(v164);
            v166 = v162 * 3;
            v167 = ulong(uint(v166)) * ulong(uint(2));
            v168 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v161) + v167);
            *reinterpret_cast<device half*>(v168) = v165;
            v169 = cm_div_approx_ftz_f32(v143, v163, cm_rcp_rom);
            v170 = half(v169);
            v171 = v166 + 1;
            v172 = ulong(uint(v171)) * ulong(uint(2));
            v173 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v161) + v172);
            *reinterpret_cast<device half*>(v173) = v170;
            v174 = 0x1p+4f;
            v175 = cm_div_approx_ftz_f32(v148, v174, cm_rcp_rom);
            v176 = half(v175);
            v177 = v166 + 2;
            v178 = ulong(uint(v177)) * ulong(uint(2));
            v179 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v161) + v178);
            *reinterpret_cast<device half*>(v179) = v176;
        }
    }
    v180 = int(v57) == int(1);
    if (v180) {
        return;
    }
    v181 = v150 == false;
    if (v181) {
    } else {
        v182 = v9 >> 4;
        v183 = cm_shared_Shared_PartialResults;
        v335 = v182 * 12;
        v184 = reinterpret_cast<threadgroup uchar*>(reinterpret_cast<threadgroup uchar*>(v183) + v335);
        *reinterpret_cast<threadgroup float*>(v184) = v138;
        v336 = reinterpret_cast<threadgroup uchar*>(reinterpret_cast<threadgroup uchar*>(v184) + 4);
        *reinterpret_cast<threadgroup float*>(v336) = v143;
        v337 = reinterpret_cast<threadgroup uchar*>(reinterpret_cast<threadgroup uchar*>(v184) + 8);
        *reinterpret_cast<threadgroup float*>(v337) = v148;
        v185 = v182 << 3;
        v186 = cm_shared_Shared_PixelPos;
        v187 = reinterpret_cast<threadgroup uchar*>(reinterpret_cast<threadgroup uchar*>(v186) + v185);
        *reinterpret_cast<threadgroup uint*>(v187) = v27;
        v338 = reinterpret_cast<threadgroup uchar*>(reinterpret_cast<threadgroup uchar*>(v187) + 4);
        *reinterpret_cast<threadgroup uint*>(v338) = v30;
    }
    v188 = v9 > 15;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (v188) {
        return;
    }
    v189 = cm_shared_Shared_PartialResults;
    v339 = v9 * 12;
    v190 = reinterpret_cast<threadgroup uchar*>(reinterpret_cast<threadgroup uchar*>(v189) + v339);
    v191 = *reinterpret_cast<threadgroup float*>(v190);
    v340 = reinterpret_cast<threadgroup uchar*>(reinterpret_cast<threadgroup uchar*>(v190) + 4);
    v192 = *reinterpret_cast<threadgroup float*>(v340);
    v341 = reinterpret_cast<threadgroup uchar*>(reinterpret_cast<threadgroup uchar*>(v190) + 8);
    v193 = *reinterpret_cast<threadgroup float*>(v341);
    v194 = v9 << 3;
    v195 = cm_shared_Shared_PixelPos;
    v196 = reinterpret_cast<threadgroup uchar*>(reinterpret_cast<threadgroup uchar*>(v195) + v194);
    v197 = *reinterpret_cast<threadgroup uint*>(v196);
    v342 = reinterpret_cast<threadgroup uchar*>(reinterpret_cast<threadgroup uchar*>(v196) + 4);
    v198 = *reinterpret_cast<threadgroup uint*>(v342);
    v199 = as_type<uint>(v191);
    v200 = simd_shuffle(v199, cm_lane_id + (uint(1) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(1) & 31u) : cm_lane_id);
    v202 = as_type<float>(v200);
    v203 = v191 + v202;
    v204 = as_type<uint>(v192);
    v205 = simd_shuffle(v204, cm_lane_id + (uint(1) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(1) & 31u) : cm_lane_id);
    v207 = as_type<float>(v205);
    v208 = v192 + v207;
    v209 = as_type<uint>(v193);
    v210 = simd_shuffle(v209, cm_lane_id + (uint(1) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(1) & 31u) : cm_lane_id);
    v212 = as_type<float>(v210);
    v213 = v193 + v212;
    v214 = as_type<uint>(v203);
    v215 = simd_shuffle(v214, cm_lane_id + (uint(2) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(2) & 31u) : cm_lane_id);
    v217 = as_type<float>(v215);
    v218 = v203 + v217;
    v219 = as_type<uint>(v208);
    v220 = simd_shuffle(v219, cm_lane_id + (uint(2) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(2) & 31u) : cm_lane_id);
    v222 = as_type<float>(v220);
    v223 = v208 + v222;
    v224 = as_type<uint>(v213);
    v225 = simd_shuffle(v224, cm_lane_id + (uint(2) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(2) & 31u) : cm_lane_id);
    v227 = as_type<float>(v225);
    v228 = v213 + v227;
    v229 = v197 | v198;
    v230 = v229 & 7;
    v231 = int(v230) != int(0);
    if (v231) {
    } else {
        v232 = v197 >> 3;
        v233 = v198 >> 3;
        v234 = v8 + 7;
        v235 = v234 >> 3;
        v236 = v1 + 7;
        v237 = v236 >> 3;
        v238 = v232 >= v235;
        v239 = v233 >= v237;
        v240 = v238 | v239;
        if (v240) {
        } else {
            v241 = v3;
            v242 = v235 * v233 + v232;
            v243 = max(float(v228), float(0x1.197998p-40f));
            v244 = cm_div_approx_ftz_f32(v218, v243, cm_rcp_rom);
            v245 = half(v244);
            v246 = v242 * 3;
            v247 = ulong(uint(v246)) * ulong(uint(2));
            v248 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v241) + v247);
            *reinterpret_cast<device half*>(v248) = v245;
            v249 = cm_div_approx_ftz_f32(v223, v243, cm_rcp_rom);
            v250 = half(v249);
            v251 = v246 + 1;
            v252 = ulong(uint(v251)) * ulong(uint(2));
            v253 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v241) + v252);
            *reinterpret_cast<device half*>(v253) = v250;
            v254 = 0x1p+6f;
            v255 = cm_div_approx_ftz_f32(v228, v254, cm_rcp_rom);
            v256 = half(v255);
            v257 = v246 + 2;
            v258 = ulong(uint(v257)) * ulong(uint(2));
            v259 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v241) + v258);
            *reinterpret_cast<device half*>(v259) = v256;
        }
    }
    v260 = int(v57) == int(2);
    if (v260) {
        return;
    }
    v261 = as_type<uint>(v218);
    v262 = simd_shuffle(v261, cm_lane_id + (uint(4) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(4) & 31u) : cm_lane_id);
    v264 = as_type<float>(v262);
    v265 = v218 + v264;
    v266 = as_type<uint>(v223);
    v267 = simd_shuffle(v266, cm_lane_id + (uint(4) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(4) & 31u) : cm_lane_id);
    v269 = as_type<float>(v267);
    v270 = v223 + v269;
    v271 = as_type<uint>(v228);
    v272 = simd_shuffle(v271, cm_lane_id + (uint(4) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(4) & 31u) : cm_lane_id);
    v274 = as_type<float>(v272);
    v275 = v228 + v274;
    v276 = as_type<uint>(v265);
    v277 = simd_shuffle(v276, cm_lane_id + (uint(8) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(8) & 31u) : cm_lane_id);
    v279 = as_type<float>(v277);
    v280 = v265 + v279;
    v281 = as_type<uint>(v270);
    v282 = simd_shuffle(v281, cm_lane_id + (uint(8) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(8) & 31u) : cm_lane_id);
    v284 = as_type<float>(v282);
    v285 = v270 + v284;
    v286 = as_type<uint>(v275);
    v287 = simd_shuffle(v286, cm_lane_id + (uint(8) & 31u) <= (cm_lane_id & (uint(31) >> 8u & 31u) | uint(31) & 31u) ? cm_lane_id + (uint(8) & 31u) : cm_lane_id);
    v289 = as_type<float>(v287);
    v290 = v275 + v289;
    v291 = v229 & 15;
    v292 = int(v291) != int(0);
    if (v292) {
        return;
    }
    v293 = v197 >> 4;
    v294 = v198 >> 4;
    v295 = v8 + 15;
    v296 = v295 >> 4;
    v297 = v1 + 15;
    v298 = v297 >> 4;
    v299 = v293 >= v296;
    v300 = v294 >= v298;
    v301 = v299 | v300;
    if (v301) {
        return;
    }
    v302 = v2;
    v303 = v296 * v294 + v293;
    v304 = max(float(v290), float(0x1.197998p-40f));
    v305 = cm_div_approx_ftz_f32(v280, v304, cm_rcp_rom);
    v306 = half(v305);
    v307 = v303 * 3;
    v308 = ulong(uint(v307)) * ulong(uint(2));
    v309 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v302) + v308);
    *reinterpret_cast<device half*>(v309) = v306;
    v310 = cm_div_approx_ftz_f32(v285, v304, cm_rcp_rom);
    v311 = half(v310);
    v312 = v307 + 1;
    v313 = ulong(uint(v312)) * ulong(uint(2));
    v314 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v302) + v313);
    *reinterpret_cast<device half*>(v314) = v311;
    v315 = 0x1p+8f;
    v316 = cm_div_approx_ftz_f32(v290, v315, cm_rcp_rom);
    v317 = half(v316);
    v318 = v307 + 2;
    v319 = ulong(uint(v318)) * ulong(uint(2));
    v320 = reinterpret_cast<device uchar*>(reinterpret_cast<device uchar*>(v302) + v319);
    *reinterpret_cast<device half*>(v320) = v317;
    return;
}
