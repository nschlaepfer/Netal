// Independent native-FG fixture renderer. Not part of the inference graph.
#include <metal_stdlib>
using namespace metal;
struct Hit { float distance; float3 point; float3 normal; float3 color; float3 previousOffset; };

void box(float3 ro, float3 rd, float3 center, float3 halfSize, float3 color, bool windows, thread Hit &hit) {
    float3 inverse = 1.0f / select(rd, float3(0.000001f), abs(rd) < 0.000001f);
    float3 lo = (center - halfSize - ro) * inverse;
    float3 hi = (center + halfSize - ro) * inverse;
    float3 near = min(lo, hi), far = max(lo, hi);
    float entry = max(near.x, max(near.y, near.z));
    float leave = min(far.x, min(far.y, far.z));
    if (entry < 0.01f || entry > leave || entry >= hit.distance) return;
    hit.distance = entry; hit.point = ro + rd * entry;
    float3 q = (hit.point - center) / halfSize;
    float3 a = abs(q);
    hit.normal = a.x > a.y && a.x > a.z ? float3(sign(q.x),0,0)
        : (a.y > a.z ? float3(0,sign(q.y),0) : float3(0,0,sign(q.z)));
    hit.color = color; hit.previousOffset = float3(0);
    if (windows && abs(hit.normal.y) < 0.5f) {
        float u = abs(hit.normal.x) > 0.5f ? hit.point.z : hit.point.x;
        float2 tile = fract(float2(u, hit.point.y) * float2(0.9f, 0.8f));
        if (tile.x > 0.18f && tile.x < 0.78f && tile.y > 0.23f && tile.y < 0.81f)
            hit.color = float3(0.075f,0.17f,0.22f);
        if (tile.x > 0.46f && tile.x < 0.49f && tile.y > 0.23f && tile.y < 0.81f)
            hit.color = color * 0.55f;
    }
}
void sphere(float3 ro, float3 rd, float3 center, float radius, float3 color, thread Hit &hit) {
    float3 oc = ro - center;
    float b = dot(oc,rd), c = dot(oc,oc) - radius * radius;
    float disc = b*b-c;
    if (disc < 0) return;
    float t = -b-sqrt(disc);
    if (t < 0.01f || t >= hit.distance) return;
    hit.distance=t; hit.point=ro+rd*t; hit.normal=normalize(hit.point-center); hit.color=color; hit.previousOffset=float3(0);
}

struct Camera { float3 origin; float3 forward; float3 right; float3 up; };
Camera cameraAt(float time, int profile) {
    float yaw = 0.12f*sin(time*0.07f);
    float3 ro = float3(0.6f*sin(time*0.055f), 1.65f, 7.0f-time*0.055f);
    if (profile == 1) { ro = float3(2.8f*sin(time*0.045f),1.8f,5.5f); yaw = -0.42f*sin(time*0.045f); }
    if (profile == 2) { yaw = 0.7f*sin(time*0.17f); ro.z=6.0f; }
    bool cut = profile == 3 && time >= 12.0f;
    if (cut) { ro=float3(-1.3f,2.1f,-2.0f); yaw=1.0f; }
    float3 forward = normalize(float3(sin(yaw),-0.035f,-cos(yaw)));
    float3 right = normalize(cross(forward,float3(0,1,0)));
    float3 up = cross(right,forward);
    return {ro, forward, right, up};
}

float4 sampleHUD(float2 uv, float2 size, float time, int profile) {
    if (profile == 4 || profile == 5) {
        float2 p = uv*size;
        return p.x>20 && p.x<100 && p.y>18 && p.y<23 ? float4(0.14f,0.85f,0.46f,1) : float4(0);
    }
    float4 ui=0;
    float2 p=uv*float2(1280,720), center=abs(p-float2(640,360));
    if ((center.x<7 && center.y<0.7f)||(center.y<7 && center.x<0.7f)) ui=float4(float3(0.96f),1);
    if(p.x>40 && p.x<225 && p.y>40 && p.y<48) ui=float4(0.14f,0.85f,0.46f,1);
    if(p.x>38 && p.x<228 && p.y>38 && p.y<50 && (p.y<40||p.y>48||p.x<40||p.x>225)) ui=float4(float3(0.92f),1);
    if(profile==6) {
        // A translucent panel over moving geometry plus crisp glyph-like bars.
        // Its changing indicator uses current UI time, independently of scene time.
        if(p.x>60 && p.x<330 && p.y>260 && p.y<380) ui=float4(float3(0.07f,0.10f,0.16f)*0.6f,0.6f);
        float2 cell=fract((p-float2(76,278))/float2(22,32));
        if(p.x>76 && p.x<294 && p.y>278 && p.y<344 && (cell.x<0.16f || cell.y<0.16f)) ui=float4(0.91f,0.94f,0.98f,1);
        float length=60+20*fmod(floor(time/3),8.0f);
        if(p.x>76 && p.x<76+length && p.y>358 && p.y<366) ui=float4(0.16f,0.89f,0.57f,1);
    }
    return ui;
}
float3 withHUD(float3 color, float2 uv, float2 size, float hudTime, int profile, bool includeHUD,
               thread float2 &motion, thread float &depth) {
    if(!includeHUD) return color;
    float4 ui=sampleHUD(uv,size,hudTime,profile);
    if(ui.a>0) { color=ui.rgb+color*(1-ui.a); motion=0; depth=0; }
    return color;
}
float3 sampleScene(float2 uv, float2 size, float time, float previousTime, float hudTime, int profile, bool includeHUD,
                   thread float2 &motion, thread float &depth) {
    // Exact planar translations isolate motion units/sign from camera/depth
    // reconstruction. The infinite pattern also avoids invented border pixels.
    if (profile == 4 || profile == 5) {
        float2 step = profile == 4 ? float2(16,0) : float2(0,16);
        float2 p = uv*size, tile = fract((p-time*step)/float2(96,72));
        float3 color = float3(0.16f,0.23f,0.31f);
        if (tile.x > 0.18f && tile.x < 0.66f && tile.y > 0.16f && tile.y < 0.70f)
            color = float3(0.78f,0.50f,0.22f);
        if (tile.x < 0.025f || tile.y < 0.033333333f) color = float3(0.90f);
        motion = (previousTime-time)*step; depth = 0.5f;
        return withHUD(color,uv,size,hudTime,profile,includeHUD,motion,depth);
    }
    if(profile==7 || profile==8) {
        float2 p=uv*size, tile=fract((p+float2(time*4,0))/float2(48,48));
        float3 color=float3(0.19f,0.34f,0.51f);
        if(tile.x<0.045f || tile.y<0.045f) color=float3(0.90f);
        float center=profile==8 ? 64+8*time : 320+120*sin(time*0.11f);
        float oldCenter=profile==8 ? 64+8*previousTime : 320+120*sin(previousTime*0.11f);
        motion=float2((time-previousTime)*4,0); depth=0.8f;
        if(abs(p.x-center)<28 && p.y>80 && p.y<280) {
            color=float3(0.75f,0.25f,0.08f);
            if(fract((p.y-80)/24)<0.09f) color=float3(0.96f);
            motion=float2(oldCenter-center,0); depth=0.2f;
        }
        return withHUD(color,uv,size,hudTime,profile,includeHUD,motion,depth);
    }
    const int cameraProfile=profile==6 ? 2 : profile;
    Camera camera = cameraAt(time, cameraProfile);
    Camera prior = cameraAt(previousTime, cameraProfile);
    float3 ro=camera.origin, forward=camera.forward, right=camera.right, up=camera.up;
    bool cut = profile == 3 && time >= 12.0f;
    float3 rd = normalize(forward + (uv.x*2.0f-1.0f)*(size.x/size.y)*0.5773502692f*right + (1.0f-uv.y*2.0f)*0.5773502692f*up);
    Hit hit = { 60.0f, float3(0), float3(0,1,0), float3(0), float3(0) };
    if (rd.y < -0.001f) {
        float t = -ro.y/rd.y;
        if (t < hit.distance) {
            hit.distance=t; hit.point=ro+rd*t;
            float2 p=hit.point.xz*1.8f;
            float2 cell=fract(p);
            float pattern=fmod(floor(p.x)+floor(p.y),2.0f)==0 ? 0.29f : 0.36f;
            if (min(cell.x,cell.y)<0.035f) pattern*=0.55f;
            hit.color=float3(pattern,pattern*0.91f,pattern*0.8f);
        }
    }
    for (int k=0;k<5;k++) {
        float z=3.0f-float(k)*4.5f;
        float3 stone=float3(0.65f,0.54f,0.40f)+(k%2)*0.08f;
        box(ro,rd,float3(-5.2f,2.8f,z),float3(1.2f,2.8f,2.0f),stone,true,hit);
        box(ro,rd,float3(5.2f,2.2f,z),float3(1.2f,2.2f,2.0f),stone*0.88f,true,hit);
    }
    // Thin poles and rails at several depths reveal parallax and flow bending.
    for(int k=0;k<14;k++) {
        float z=4.0f-float(k)*0.9f;
        box(ro,rd,float3(-2.7f,0.62f,z),float3(0.035f,0.62f,0.035f),float3(0.1f),false,hit);
    }
    box(ro,rd,float3(-2.7f,1.12f,-1.8f),float3(0.045f,0.025f,6.0f),float3(0.1f),false,hit);
    box(ro,rd,float3(-2.7f,0.45f,-1.8f),float3(0.045f,0.025f,6.0f),float3(0.1f),false,hit);
    // Independently moving simple figures, with silhouettes and occlusion.
    for(int k=0;k<2;k++) {
        float x=1.6f*sin(time*0.10f+float(k)*2.3f), z=-1.0f-float(k)*3.4f;
        float3 coat=k==0 ? float3(0.62f,0.12f,0.09f) : float3(0.08f,0.24f,0.60f);
        float oldX=1.6f*sin(previousTime*0.10f+float(k)*2.3f);
        float before=hit.distance;
        box(ro,rd,float3(x,0.8f,z),float3(0.22f,0.52f,0.18f),coat,false,hit);
        if(hit.distance<before) hit.previousOffset=float3(oldX-x,0,0);
        before=hit.distance;
        sphere(ro,rd,float3(x,1.48f,z),0.17f,float3(0.73f,0.53f,0.38f),hit);
        if(hit.distance<before) hit.previousOffset=float3(oldX-x,0,0);
        for(int leg=0;leg<2;leg++) {
            float step=0.1f*sin(time*0.6f+float(leg)*3.14159f);
            float oldStep=0.1f*sin(previousTime*0.6f+float(leg)*3.14159f);
            before=hit.distance;
            box(ro,rd,float3(x+(leg==0 ? -0.1f : 0.1f),0.22f,z+step),float3(0.075f,0.22f,0.08f),float3(0.12f),false,hit);
            if(hit.distance<before) hit.previousOffset=float3(oldX-x,0,oldStep-step);
        }
    }
    float3 sky=float3(0.24f,0.40f,0.58f);
    float3 color=sky;
    if(hit.distance<60.0f) {
        float lighting=0.38f+0.62f*max(0.0f,dot(hit.normal,normalize(float3(-0.5f,1.0f,0.4f))));
        color=mix(hit.color*lighting,sky,clamp(hit.distance*0.008f,0.0f,0.4f));
    }
    if(cut) color=clamp(color*0.45f+float3(0.48f,0.37f,0.50f),0.0f,1.0f);
    // Current-to-previous motion in pixels. Rigid figure parts are projected
    // from their exact previous world positions; static geometry is unchanged.
    // Sky uses rotation-only reprojection and forward depth 1.
    float3 previousPoint = hit.distance < 60 ? hit.point + hit.previousOffset : prior.origin + rd*60;
    float3 delta = previousPoint - prior.origin;
    float priorZ = dot(delta, prior.forward);
    if(priorZ>0.001f) {
        float2 previousUV=float2(0.5f+0.5f*dot(delta,prior.right)/(priorZ*0.5773502692f*(size.x/size.y)),
                                 0.5f-0.5f*dot(delta,prior.up)/(priorZ*0.5773502692f));
        motion=(previousUV-uv)*size;
    } else motion=float2(65504.0f); // pinned NGX invalid motion sentinel
    float z=hit.distance<60 ? dot(hit.point-ro,forward) : 1000.0f;
    depth=clamp(1000.0f/999.9f-(1000.0f*0.1f)/(999.9f*max(z,0.1f)),0.0f,1.0f);
    return withHUD(color,uv,size,hudTime,profile,includeHUD,motion,depth);
}
kernel void simulate_native(texture2d<float,access::write> color [[texture(0)]],
    texture2d<half,access::write> motionOut [[texture(1)]],
    texture2d<float,access::write> depthOut [[texture(2)]],
    texture2d<float,access::write> hudOut [[texture(3)]],
    constant float4 &parameters [[buffer(0)]], constant uint &includeHUD [[buffer(1)]],
    constant float &hudTime [[buffer(2)]], uint2 xy [[thread_position_in_grid]]) {
    if(xy.x>=color.get_width()||xy.y>=color.get_height()) return;
    float2 size=float2(color.get_width(),color.get_height());
    float2 motion=0; float depth=1;
    float2 center=(float2(xy)+0.5f)/size;
    sampleScene(center,size,parameters.x,parameters.y,hudTime,int(parameters.z),includeHUD!=0,motion,depth);
    // Color gets coverage AA. Geometry/motion/depth retain the center sample's
    // exact surface identity, avoiding blends between foreground and background.
    float3 rgb=0; float4 hud=0;
    for(int y=0;y<2;y++) for(int x=0;x<2;x++) {
        float2 ignoredMotion; float ignoredDepth;
        float2 uv=(float2(xy)+float2(0.25f+0.5f*x,0.25f+0.5f*y))/size;
        rgb+=sampleScene(uv,size,parameters.x,parameters.y,hudTime,int(parameters.z),includeHUD!=0,ignoredMotion,ignoredDepth);
        hud+=sampleHUD(uv,size,hudTime,int(parameters.z));
    }
    color.write(float4(rgb*0.25f,1),xy);
    motionOut.write(half4(half2(parameters.w>0.5f ? float2(0) : motion),half(0),half(0)),xy);
    depthOut.write(float4(depth,0,0,0),xy);
    hudOut.write(hud*0.25f,xy);
}
