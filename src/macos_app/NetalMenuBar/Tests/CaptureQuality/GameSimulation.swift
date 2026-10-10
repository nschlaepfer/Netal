import Foundation
import Metal

/// An independent procedural renderer, not a game capture or an interpolation of
/// endpoint images. Rendering at any fractional time supplies reference motion,
/// parallax and occlusion without giving those scene inputs to Netal's processor.
final class GameSimulation {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw Failure(description: "Simulation queue unavailable") }
        self.queue = queue
        let library = try device.makeLibrary(source: Self.shader, options: nil)
        guard let function = library.makeFunction(name: "simulate") else { throw Failure(description: "Simulation shader missing") }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    func render(profile: Int, time: Double, width: Int, height: Int) throws -> [UInt8] {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
            width: width, height: height, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        guard let texture = device.makeTexture(descriptor: descriptor),
              let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw Failure(description: "Simulation resources unavailable")
        }
        var parameters = SIMD4<Float>(Float(time), Float(profile), 0, 0)
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        encoder.dispatchThreads(.init(width: width, height: height, depth: 1),
            threadsPerThreadgroup: .init(width: 16, height: 8, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure(description: "Simulation rendering failed") }
        var result = [UInt8](repeating: 0, count: width * height * 4)
        result.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!, bytesPerRow: width * 4,
                from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        return result
    }

    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;
    struct Hit { float distance; float3 point; float3 normal; float3 color; };

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
        hit.color = color;
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
        hit.distance=t; hit.point=ro+rd*t; hit.normal=normalize(hit.point-center); hit.color=color;
    }

    float3 sampleScene(float2 uv, float aspect, float time, int profile) {
        float yaw = 0.12f*sin(time*0.07f);
        float3 ro = float3(0.6f*sin(time*0.055f), 1.65f, 7.0f-time*0.055f);
        if (profile == 1) { ro = float3(2.8f*sin(time*0.045f),1.8f,5.5f); yaw = -0.42f*sin(time*0.045f); }
        if (profile == 2) { yaw = 0.7f*sin(time*0.17f); ro.z=6.0f; }
        bool cut = profile == 3 && time >= 12.0f;
        if (cut) { ro=float3(-1.3f,2.1f,-2.0f); yaw=1.0f; }
        float3 forward = normalize(float3(sin(yaw),-0.035f,-cos(yaw)));
        float3 right = normalize(cross(forward,float3(0,1,0)));
        float3 up = cross(right,forward);
        float3 rd = normalize(forward + (uv.x*2.0f-1.0f)*aspect*0.66f*right + (1.0f-uv.y*2.0f)*0.66f*up);
        Hit hit = { 60.0f, float3(0), float3(0,1,0), float3(0) };
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
            box(ro,rd,float3(x,0.8f,z),float3(0.22f,0.52f,0.18f),coat,false,hit);
            sphere(ro,rd,float3(x,1.48f,z),0.17f,float3(0.73f,0.53f,0.38f),hit);
            for(int leg=0;leg<2;leg++) {
                float step=0.1f*sin(time*0.6f+float(leg)*3.14159f);
                box(ro,rd,float3(x+(leg==0 ? -0.1f : 0.1f),0.22f,z+step),float3(0.075f,0.22f,0.08f),float3(0.12f),false,hit);
            }
        }
        float3 sky=float3(0.24f,0.40f,0.58f);
        float3 color=sky;
        if(hit.distance<60.0f) {
            float lighting=0.38f+0.62f*max(0.0f,dot(hit.normal,normalize(float3(-0.5f,1.0f,0.4f))));
            color=mix(hit.color*lighting,sky,clamp(hit.distance*0.008f,0.0f,0.4f));
        }
        if(cut) color=clamp(color*0.45f+float3(0.48f,0.37f,0.50f),0.0f,1.0f);
        // A screen-fixed reticle and compact health bar are independent of camera motion.
        float2 p=uv*float2(1280,720), center=abs(p-float2(640,360));
        if ((center.x<7 && center.y<0.7f)||(center.y<7 && center.x<0.7f)) color=float3(0.96f);
        if(p.x>40 && p.x<225 && p.y>40 && p.y<48) color=float3(0.14f,0.85f,0.46f);
        if(p.x>38 && p.x<228 && p.y>38 && p.y<50 && (p.y<40||p.y>48||p.x<40||p.x>225)) color=float3(0.92f);
        return color;
    }
    kernel void simulate(texture2d<float,access::write> output [[texture(0)]],
        constant float4 &parameters [[buffer(0)]], uint2 xy [[thread_position_in_grid]]) {
        if(xy.x>=output.get_width()||xy.y>=output.get_height()) return;
        float2 size=float2(output.get_width(),output.get_height());
        float3 color=0;
        // Fixed four-sample coverage AA is shared by source/reference rasterization,
        // never by the processor under test. No motion vectors/depth are supplied.
        for(int y=0;y<2;y++) for(int x=0;x<2;x++)
            color+=sampleScene((float2(xy)+float2(0.25f+0.5f*x,0.25f+0.5f*y))/size,
                size.x/size.y,parameters.x,int(parameters.y));
        output.write(float4(color*0.25f,1),xy);
    }
    """
}
