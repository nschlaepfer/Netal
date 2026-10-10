#!/usr/bin/env python3
"""Build a pinned 85/85 standalone FG launch-coverage certificate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HASH = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_SHA256 = "5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b"


class CoverageError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CoverageError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode())


@dataclass(frozen=True)
class Unit:
    name: str
    launches: tuple[int, ...]
    path: str
    file_sha256: str
    root_field: str | None
    evidence_root_sha256: str


UNITS = (
    Unit("early color/history module11", (3, 13), "evidence/specs/fg-early-l3-l13-v1/verification.json", "18317d008f591fc20ce3d61a28ab9931b5e36d978e12898c3fa98102fb3df876", "receipt_root_sha256", "e9b83e2b2bd128b71b5a19fc3b769c75511903c3a8c4f0dc0857c8701a86a599"),
    Unit("early vector/index module15", (4, 14), "evidence/specs/fg-early-l4-l14-v1/accepted-receipt.json", "ac1f1cd58def89831095711d49ab51273c8d3a8b8567980ffbb69cd7e3b6e9e1", "receipt_root_sha256", "c384c63300a70d64fd120296d1c7f8ba6682f179184439505166490cfc716276"),
    Unit("early atomic seed module13", (5, 15), "evidence/specs/fg-early-l5-l15-v1/accepted-receipt.json", "987531c8e50abc1b80387a21492fe8da4e6c90d7955c1dc60abcfc0ecb4995dc", "receipt_root_sha256", "39c1421e2cb0728b1e7fbc536e506da4bb58fc4ccb50f2c89ae8603e08898126"),
    Unit("early multiscale producer module21", (6, 16), "evidence/specs/fg-early-l6-l16-v1/accepted-receipt.json", "78b19d1a8120ab6be56a27bb10f6f15815c77772e32821d5ad951e307f72f115", "receipt_root_sha256", "f963c835db6a49aaff6d75ce3e96da555f9a9650e5b8af15d3990776a53109de"),
    Unit("early multiscale repair module23", (7, 17), "evidence/specs/fg-early-l7-l17-v1/accepted-receipt.json", "789807c13c5966f98bfde7fd0fa1a7a66c7d9c32addb649adad2267e1c24e9cf", "receipt_root_sha256", "3c9fc224ddfd16e8d83f1d745f76a1c5d16b8d93591088db466e5f2d99b1ec36"),
    Unit("early scatter mask module25", (8, 18), "evidence/specs/fg-early-l8-l18-v1/accepted-receipt.json", "559a8d0023f935f0ee1de3b6d0b11cbda827f4c104ebdb4635b71a64fbd8a3d0", "receipt_root_sha256", "b9945e4ff78fa791d50bdf17df3d3ed677ec75c1ba35f04c1fca5167a21060bc"),
    Unit("early RGB projection module14", (9,), "evidence/specs/fg-early-l9-v1/accepted-receipt.json", "0d3dfef8a999eec90ca045c8609dce2c0ac8400bf6b7532ad7b56a3ed828c51e", "receipt_root_sha256", "6e69cbea80011a0db7bdac95cac3e2a6bdd34bbd98974990fcde29114cabf8a2"),
    Unit("optional status module27", (10, 12), "evidence/specs/fg-early-status-l10-l12-v1/accepted-receipt.json", "e5c44c0536a62e5c95fcab7eae86828d2728f2a3928b6ff2e57f43bc00cdc454", "receipt_root_sha256", "2459cd20908178e277aa74e6ed49228c11ce8d76d5f2c57eda4da301d9b35698"),
    Unit("reset presentation copy module4", (11,), "evidence/specs/fg-early-l11-v1/accepted-receipt.json", "10b722579beab430bade6a908cdf696c16119b13b613467ae5b2f508987fb9fa", "receipt_root_sha256", "941fe93d35ed544ece657730e3846ff34a698711ec094ee439d2a2552e0618aa"),
    Unit("early feature assembler module5", (19,), "evidence/specs/fg-early-l19-v1/accepted-receipt.json", "d8d21cf5051e69d5dbf6660a51fa6e431e2b2d2be9191950a7c80de4c4d5827c", "receipt_root_sha256", "5acda2522c0e9465c6ccce26676ebd3bd0d29f1ce7aa936e11ec03bec19db4e3"),
    Unit("generic convolution family", (20, 22, 24, 26, 28, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51), "evidence/specs/fg-conv-family-v1/verification.json", "f973de2f4669ad9f94b7c967c583c75c18c83bfe19b15323e7f152c946fa3a66", "receipt_root_sha256", "b01e03d08e309049ef89ed489b4051213a455b952182068492fe9ec5443a87fd"),
    Unit("pooling family", (21, 23, 25, 27, 29), "evidence/specs/fg-pooling-v1/receipt.json", "54911bf54560ff96bd8c3bd1da363e89b6e78a2f1d6f0cc35e2122313c9616b0", "evidence_root_sha256", "535542da2498562342ff8598812ca2fb5526e4de6d4e43b47764878db6cdd557"),
    Unit("upscale family", (32, 36, 40, 44, 48), "evidence/specs/fg-upscale-v1/accepted-receipt.json", "10b0c7825a47558238374694c8978cf2c5feb5c0c337538c89bb3c272dbe3113", "verification_root_sha256", "7bd3a7baca23fe8c1a3befd103794669780abf3acc8a97d80cf9fb18e218a178"),
    Unit("elementwise family", (33, 37, 41, 45, 49), "evidence/specs/fg-elementwise-family-v2/accepted-receipt.json", "fe4bba83d5d3ed61f390a498a48fc7691d423375b3496e5de9c7aac2c39ec5da", "evidence_root_sha256", "5fff4f8d0fc54da2ed389dbb908d9b45ea86eabfed8beac79024692ab70b8dc0"),
    Unit("middle depth-aware module6", (52,), "evidence/specs/fg-mid-l52-v1/accepted-receipt.json", "c399dca6f4aa0edded7d1767521d8a2d52854150116450c54bcc242ea21bf484", "receipt_root_sha256", "e49521b88b7a3b603bd9a605a523e059ba417ccd9d57a382bd56d7ac142740b6"),
    Unit("middle motion assignment module12", (53,), "evidence/specs/fg-mid-l53-v1/accepted-receipt.json", "dd7e5379210635e5550bbde8faf5f0b1a3890ee68d57ce764168b00a781b8142", "receipt_root_sha256", "b2ae779aa98cf37e4836332f9f86148e4c583380a1c462dbb058050c48fa7480"),
    Unit("middle confidence module26", (54,), "evidence/specs/fg-mid-l54-v1/accepted-receipt.json", "ff32aeb81ce0b09977127f3f6dc0d0274d46530233d099635b97c809d463f159", "receipt_root_sha256", "e27a723f2be1e55232ea1b3c9ab2728fe1799db63ad11786c63e573efa3a5a53"),
    Unit("middle six-output module3", (55,), "evidence/specs/fg-mid-l55-v1/accepted-receipt.json", "b2f249b5cd6a3f125b61997c53e9be2f7f20dbef38326e078cfecd48f0da085e", "receipt_root_sha256", "0c27feb46489bc29a774519ce0769d10cc4f663d4ef6a4f0d2bcaf2adf5db8ce"),
    Unit("initial merge module53", (56,), "evidence/specs/fg-initial-merge-v1/accepted-receipt.json", "16eb85909f7110c48f841ccee8e83bafe9e80ba80bf228f68c6bcc95628bcb24", "evidence_root_sha256", "4f22ce06ecdec628c2b90f4a8644ae38bfd36b71ca693d5a34b511a0e91b675c"),
    Unit("custom convolution aggregate", (57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83), "evidence/specs/fg-custom-conv-25-aggregate-v1/aggregate.json", "75e8d03980183c4a6741b8e12d6e81c169bf994596b14bedfe641dce1c815ce6", "receipt_root_sha256", "6309c8c32c7fcb17434f429f16220b2385cfe6e8f34dfd329d5abede3641e1c0"),
    Unit("custom upsample module60", (70,), "evidence/specs/fg-custom-upsample-v1/accepted-receipt.json", "785375251e710208852f2da817ddcd73aeba96c49d0479a13fe03383fdb3794f", "verification_root_sha256", "2d080a5eeaed4f9be08f6663793561c889e785056c7e221e93863ff7c60cfc40"),
    Unit("central block module61", (71,), "evidence/specs/fg-central-block-v1/accepted-receipt.json", "3e905ecc23ea8ed51a0c7b4393eddf607793bc702a09ad71a3598724e2f108ee", None, "3e905ecc23ea8ed51a0c7b4393eddf607793bc702a09ad71a3598724e2f108ee"),
    Unit("postprocess module17", (84,), "evidence/specs/fg-postprocess-l84-v1/accepted-receipt.json", "518e6f1517d5f6909d1de61a3d1af38aec4fb46d43fb4fb9154aea293519a7e8", "receipt_root_sha256", "a4c067f820fa40d3d3d31ab5c4294b96349095b56e34db043da2dbeff75cd152"),
    Unit("postprocess module18", (85,), "evidence/specs/fg-postprocess-l85-v1/accepted-receipt.json", "2b1fa6f0f21022004f482398b5b54e8cd3f24e7cf0fc911e7823e96b254719bd", "receipt_root_sha256", "5ba99aa4a99358d39f793263c1215688ee2ad451f0820440e62c820cded98029"),
    Unit("postprocess module19", (86,), "evidence/specs/fg-postprocess-l86-v1/accepted-receipt.json", "22eb5fdce8234fa26b24a94e8345dbfebc8b77e7bbb45002b7db54717b85b7ce", "receipt_root_sha256", "8072e544a9453a7c053fc5d784855a339af39e4c0a81c88a74a6746758252af6"),
    Unit("final compositor module20", (87,), "evidence/specs/fg-compositor-l87-v1/accepted-receipt.json", "3f41d714d81a83c0760a9cf4cf8114a7d315946afac7c491b268df3ff10995ad", "receipt_root_sha256", "fb10d59c834cfabcbff7a1422a6e56a9bd91dcb92f67271e66b18b22c560792e"),
)


def validate_certificate(value: dict[str, Any]) -> None:
    require(value.get("standalone_launch_count") == 85, "standalone launch count drift")
    require(value.get("covered_launches") == list(range(3, 88)), "launch coverage drift")
    require(value.get("duplicate_launches") == [], "duplicate launch coverage")
    require(value.get("missing_launches") == [], "missing launch coverage")
    claims = value.get("claims", {})
    require(claims.get("all_640x360_selected_launches_exact_standalone") is True, "standalone claim missing")
    for name in (
        "connected_l3_l87_parity",
        "temporal_metal_parity",
        "large_resolution_metal_parity",
        "feature_mode_parity",
        "netal_integration",
        "performance",
    ):
        require(claims.get(name) is False, f"certificate overclaims {name}")
    require(value.get("address_fields_recorded") is False, "certificate contains addresses")


def negative_audit(certificate: dict[str, Any]) -> dict[str, Any]:
    cases = []

    def reject(name: str, mutation: Any) -> None:
        value = copy.deepcopy(certificate)
        mutation(value)
        rejected = False
        try:
            validate_certificate(value)
        except CoverageError:
            rejected = True
        require(rejected, f"coverage mutation accepted: {name}")
        cases.append({"case": name, "rejected": True})

    reject("count", lambda value: value.__setitem__("standalone_launch_count", 84))
    reject("coverage", lambda value: value["covered_launches"].pop())
    reject("duplicate", lambda value: value["duplicate_launches"].append(19))
    reject("missing", lambda value: value["missing_launches"].append(19))
    reject("standalone", lambda value: value["claims"].__setitem__("all_640x360_selected_launches_exact_standalone", False))
    reject("connected-overclaim", lambda value: value["claims"].__setitem__("connected_l3_l87_parity", True))
    reject("temporal-overclaim", lambda value: value["claims"].__setitem__("temporal_metal_parity", True))
    reject("resolution-overclaim", lambda value: value["claims"].__setitem__("large_resolution_metal_parity", True))
    reject("feature-overclaim", lambda value: value["claims"].__setitem__("feature_mode_parity", True))
    reject("netal-overclaim", lambda value: value["claims"].__setitem__("netal_integration", True))
    reject("performance-overclaim", lambda value: value["claims"].__setitem__("performance", True))
    reject("addresses", lambda value: value.__setitem__("address_fields_recorded", True))
    result = {
        "schema": "neuralstudio.fg.standalone-launch-coverage-negative-audit.v1",
        "case_count": len(cases),
        "rejected_count": len(cases),
        "cases": cases,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical_sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()

    seen: dict[int, str] = {}
    duplicates = []
    receipts = []
    for unit in UNITS:
        path = repository / unit.path
        require(path.is_file() and not path.is_symlink(), f"evidence missing: {unit.path}")
        require(sha_file(path) == unit.file_sha256, f"evidence file drift: {unit.path}")
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        require(isinstance(value, dict), f"evidence must be an object: {unit.path}")
        require(value.get("result") == "pass", f"evidence rejected: {unit.path}")
        if unit.root_field is not None:
            require(value.get(unit.root_field) == unit.evidence_root_sha256, f"evidence root drift: {unit.path}")
        require(HASH.fullmatch(unit.evidence_root_sha256) is not None, "invalid evidence root")
        for launch in unit.launches:
            if launch in seen:
                duplicates.append(launch)
            else:
                seen[launch] = unit.name
        receipts.append({
            "unit": unit.name,
            "launches": list(unit.launches),
            "evidence_path": unit.path,
            "evidence_file_sha256": unit.file_sha256,
            "evidence_root_sha256": unit.evidence_root_sha256,
        })

    covered = sorted(seen)
    expected = list(range(3, 88))
    missing = sorted(set(expected).difference(covered))
    require(not duplicates and not missing and covered == expected, "standalone launch partition is incomplete")
    certificate = {
        "schema": "neuralstudio.fg.standalone-launch-coverage.v1",
        "pinned_runtime_sha256": RUNTIME_SHA256,
        "extent": {"width": 640, "height": 360},
        "canonical_launch_range": [3, 87],
        "standalone_launch_count": len(covered),
        "covered_launches": covered,
        "duplicate_launches": duplicates,
        "missing_launches": missing,
        "unit_count": len(receipts),
        "units": receipts,
        "address_fields_recorded": False,
        "claims": {
            "all_640x360_selected_launches_exact_standalone": True,
            "connected_l3_l87_parity": False,
            "temporal_metal_parity": False,
            "large_resolution_metal_parity": False,
            "feature_mode_parity": False,
            "netal_integration": False,
            "performance": False,
        },
        "result": "pass",
    }
    validate_certificate(certificate)
    certificate["receipt_root_sha256"] = canonical_sha(certificate)
    audit = negative_audit(certificate)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    arguments.negative_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"covered={len(covered)}/85")
    print(f"receipt_root_sha256={certificate['receipt_root_sha256']}")
    print(f"negative_root_sha256={audit['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
