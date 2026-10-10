struct NSFGTailResourceArena {
  id<MTLDevice> device = nil;
  std::vector<id<MTLComputePipelineState>> pipelines;
  std::vector<std::string> pipeline_keys;
  std::vector<id<MTLBuffer>> guarded_buffers;
  std::vector<std::size_t> guarded_sizes;
  std::vector<id<MTLBuffer>> raw_buffers;
  std::vector<std::size_t> raw_sizes;
  std::vector<id<MTLTexture>> connected_textures;
  std::vector<id<MTLTexture>> tail_textures;
  std::size_t pipeline_cursor = 0;
  std::size_t guarded_cursor = 0;
  std::size_t raw_cursor = 0;
  std::size_t connected_cursor = 0;
  std::size_t tail_cursor = 0;
  std::uint64_t pass_allocations = 0;

  void Begin(id<MTLDevice> requested_device) {
    if (requested_device == nil)
      throw std::runtime_error("runtime tail device is not bound");
    if (device != nil && device != requested_device)
      throw std::runtime_error("runtime tail cache cannot cross Metal devices");
    device = requested_device;
    pipeline_cursor = guarded_cursor = raw_cursor = connected_cursor =
        tail_cursor = 0;
    pass_allocations = 0;
  }

  void Clear() {
    pipelines.clear();
    pipeline_keys.clear();
    guarded_buffers.clear();
    guarded_sizes.clear();
    raw_buffers.clear();
    raw_sizes.clear();
    connected_textures.clear();
    tail_textures.clear();
    pipeline_cursor = guarded_cursor = raw_cursor = connected_cursor =
        tail_cursor = 0;
    pass_allocations = 0;
    device = nil;
  }
};

NSFGTailResourceArena& NSFGTailArena() {
  static auto* arena = new NSFGTailResourceArena();
  return *arena;
}

void NSFGTailBeginResourcePass(id<MTLDevice> device) {
  NSFGTailArena().Begin(device);
}

std::uint64_t NSFGTailPassResourceAllocations() {
  return NSFGTailArena().pass_allocations;
}

void Wait(id<MTLCommandBuffer> command) {
  NSFGWaitAndRecord(command, NSFG_SPAN_TAIL);
}

void NSFGTailReleasePersistentResourcesImpl() { NSFGTailArena().Clear(); }

id<MTLComputePipelineState> Pipeline(id<MTLDevice> device,
                                     const std::filesystem::path& path,
                                     const char* name) {
  auto& arena = NSFGTailArena();
  const auto index = arena.pipeline_cursor++;
  const std::string key = path.string() + "#" + name;
  if (index == arena.pipelines.size()) {
    arena.pipelines.push_back(NSFGAcceptedPipeline(device, path, name));
    arena.pipeline_keys.push_back(key);
    ++arena.pass_allocations;
  } else if (arena.pipeline_keys[index] != key) {
    throw std::runtime_error("runtime tail pipeline order drift");
  }
  return arena.pipelines[index];
}

id<MTLBuffer> Guarded(id<MTLDevice> device,
                      const std::vector<std::uint8_t>* initial,
                      std::size_t bytes, std::uint8_t poison) {
  auto& arena = NSFGTailArena();
  const auto index = arena.guarded_cursor++;
  if (index == arena.guarded_buffers.size()) {
    arena.guarded_buffers.push_back(
        NSFGAcceptedGuarded(device, initial, bytes, poison));
    arena.guarded_sizes.push_back(bytes);
    ++arena.pass_allocations;
  } else {
    if (arena.guarded_sizes[index] != bytes)
      throw std::runtime_error("runtime tail guarded-buffer order drift");
    if (!NSFGTailAuditEnabled()) return arena.guarded_buffers[index];
    auto buffer = arena.guarded_buffers[index];
    std::memset(buffer.contents, kGuardValue, buffer.length);
    auto* payload = static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes;
    if (initial != nullptr) {
      if (initial->size() != bytes)
        throw std::runtime_error("runtime tail initial buffer size drift");
      std::memcpy(payload, initial->data(), bytes);
    } else {
      std::memset(payload, poison, bytes);
    }
  }
  return arena.guarded_buffers[index];
}

id<MTLBuffer> NSFGCachedRawBuffer(id<MTLDevice> device, const void* bytes,
                                  std::size_t size) {
  auto& arena = NSFGTailArena();
  const auto index = arena.raw_cursor++;
  if (index == arena.raw_buffers.size()) {
    id<MTLBuffer> buffer = [device newBufferWithBytes:bytes
                                               length:size
                                              options:MTLResourceStorageModeShared];
    if (buffer == nil) throw std::runtime_error("runtime tail parameter buffer");
    arena.raw_buffers.push_back(buffer);
    arena.raw_sizes.push_back(size);
    ++arena.pass_allocations;
  } else {
    if (arena.raw_sizes[index] != size)
      throw std::runtime_error("runtime tail parameter-buffer order drift");
    if (NSFGTailAuditEnabled())
      std::memcpy(arena.raw_buffers[index].contents, bytes, size);
  }
  return arena.raw_buffers[index];
}

id<MTLTexture> ConnectedTexture(id<MTLDevice> device, MTLPixelFormat format,
                                NSUInteger width, NSUInteger height,
                                MTLTextureUsage usage,
                                const std::vector<std::uint8_t>& initial,
                                NSUInteger bytes_per_pixel) {
  auto& arena = NSFGTailArena();
  const auto index = arena.connected_cursor++;
  if (index == arena.connected_textures.size()) {
    arena.connected_textures.push_back(NSFGAcceptedConnectedTexture(
        device, format, width, height, usage, initial, bytes_per_pixel));
    ++arena.pass_allocations;
  } else {
    auto texture = arena.connected_textures[index];
    if (texture.pixelFormat != format || texture.width != width ||
        texture.height != height || texture.usage != usage ||
        initial.size() != width * height * bytes_per_pixel)
      throw std::runtime_error("runtime tail connected-texture order drift");
    if (!NSFGTailAuditEnabled()) return texture;
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
               mipmapLevel:0
                 withBytes:initial.data()
               bytesPerRow:width * bytes_per_pixel];
  }
  return arena.connected_textures[index];
}

id<MTLTexture> TailTexture(id<MTLDevice> device, MTLPixelFormat format,
                           MTLTextureUsage usage, std::uint8_t poison,
                           std::size_t bytes_per_pixel) {
  auto& arena = NSFGTailArena();
  const auto index = arena.tail_cursor++;
  if (index == arena.tail_textures.size()) {
    arena.tail_textures.push_back(
        NSFGAcceptedTailTexture(device, format, usage, poison,
                                bytes_per_pixel));
    ++arena.pass_allocations;
  } else {
    auto texture = arena.tail_textures[index];
    if (texture.pixelFormat != format || texture.width != 640 ||
        texture.height != 360 || texture.usage != usage)
      throw std::runtime_error("runtime tail texture order drift");
    if (!NSFGTailAuditEnabled()) return texture;
    const std::vector<std::uint8_t> initial(
        640u * 360u * bytes_per_pixel, poison);
    [texture replaceRegion:MTLRegionMake2D(0, 0, 640, 360)
               mipmapLevel:0
                 withBytes:initial.data()
               bytesPerRow:640u * bytes_per_pixel];
  }
  return arena.tail_textures[index];
}
