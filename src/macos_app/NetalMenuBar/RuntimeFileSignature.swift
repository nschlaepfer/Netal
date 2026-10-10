import Foundation
import Darwin

/// Metadata used to invalidate cached integrity checks, not a substitute for hashing.
struct RuntimeFileSignature: Equatable {
    let path: String
    let inode: UInt64
    let device: Int32
    let mode: UInt16
    let size: Int64
    let modifiedSeconds: Int
    let modifiedNanoseconds: Int
    let changedSeconds: Int
    let changedNanoseconds: Int
    var regular: Bool { mode & UInt16(S_IFMT) == UInt16(S_IFREG) }
    var directory: Bool { mode & UInt16(S_IFMT) == UInt16(S_IFDIR) }

    static func read(_ url: URL) -> RuntimeFileSignature? {
        var info = stat()
        guard lstat(url.path, &info) == 0 else { return nil }
        return RuntimeFileSignature(path: url.path, inode: info.st_ino, device: info.st_dev,
                                    mode: info.st_mode, size: info.st_size,
                                    modifiedSeconds: info.st_mtimespec.tv_sec,
                                    modifiedNanoseconds: info.st_mtimespec.tv_nsec,
                                    changedSeconds: info.st_ctimespec.tv_sec,
                                    changedNanoseconds: info.st_ctimespec.tv_nsec)
    }
}
