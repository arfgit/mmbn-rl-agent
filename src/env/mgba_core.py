import ctypes
import ctypes.util
import struct
from pathlib import Path

import numpy as np

_LIB_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent / '.venv' / 'lib' / 'libmgba.0.11.dylib',
    Path('/opt/homebrew/lib/libmgba.dylib'),
    Path('/usr/local/lib/libmgba.dylib'),
]

_lib = None
for _path in _LIB_CANDIDATES:
    if _path.exists():
        _lib = ctypes.cdll.LoadLibrary(str(_path))
        break
if _lib is None:
    raise RuntimeError("libmgba not found. Install with: brew install mgba")

_lib.GBACoreCreate.restype = ctypes.c_void_p
_lib.GBACoreCreate.argtypes = []
_lib.mCoreLoadFile.restype = ctypes.c_bool
_lib.mCoreLoadFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.mCoreInitConfig.restype = None
_lib.mCoreInitConfig.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.mCoreLoadSaveFile.restype = ctypes.c_bool
_lib.mCoreLoadSaveFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_lib.mCoreLoadState.restype = ctypes.c_bool
_lib.mCoreLoadState.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.mCoreSaveState.restype = ctypes.c_bool
_lib.mCoreSaveState.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.mCoreLoadStateNamed.restype = ctypes.c_bool
_lib.mCoreLoadStateNamed.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_lib.mCoreSaveStateNamed.restype = ctypes.c_bool
_lib.mCoreSaveStateNamed.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

SAVESTATE_ALL = 31

GBA_W, GBA_H = 240, 160

GBA_KEY_A = 0
GBA_KEY_B = 1
GBA_KEY_SELECT = 2
GBA_KEY_START = 3
GBA_KEY_RIGHT = 4
GBA_KEY_LEFT = 5
GBA_KEY_UP = 6
GBA_KEY_DOWN = 7
GBA_KEY_R = 8
GBA_KEY_L = 9

KEY_NAMES = ['A', 'B', 'SELECT', 'START', 'RIGHT', 'LEFT', 'UP', 'DOWN', 'R', 'L']

_VT = {
    'init': 0, 'deinit': 1, 'platform': 2, 'supportsFeature': 3,
    'setSync': 4, 'loadConfig': 5, 'reloadConfigOption': 6, 'setOverride': 7,
    'baseVideoSize': 8, 'currentVideoSize': 9, 'videoScale': 10,
    'screenRegions': 11, 'setVideoBuffer': 12, 'setVideoGLTex': 13,
    'getPixels': 14, 'putPixels': 15, 'audioSampleRate': 16,
    'getAudioBuffer': 17, 'setAudioBufferSize': 18, 'getAudioBufferSize': 19,
    'addCoreCallbacks': 20, 'clearCoreCallbacks': 21, 'setAVStream': 22,
    'isROM': 23, 'loadROM': 24, 'loadSave': 25, 'loadTemporarySave': 26,
    'unloadROM': 27, 'romSize': 28, 'checksum': 29, 'loadBIOS': 30,
    'selectBIOS': 31, 'loadPatch': 32,
    'reset': 33, 'runFrame': 34, 'runLoop': 35, 'step': 36,
    'stateSize': 37, 'loadState': 38, 'saveState': 39,
    'loadExtraState': 40, 'saveExtraState': 41,
    'setKeys': 42, 'addKeys': 43, 'clearKeys': 44, 'getKeys': 45,
    'frameCounter': 46, 'frameCycles': 47, 'frequency': 48,
    'getGameInfo': 49, 'setPeripheral': 50, 'getPeripheral': 51,
    'busRead8': 52, 'busRead16': 53, 'busRead32': 54,
}


def _find_vtable_offset(core_ptr: int) -> int:
    raw = bytes((ctypes.c_ubyte * 4096).from_address(core_ptr))
    cur_start = -1
    cur_len = 0
    for offset in range(0, 4096, 8):
        val = struct.unpack_from('<Q', raw, offset)[0]
        if val != 0 and 0x100000000 <= val <= 0x7fffffffffff:
            if cur_start < 0:
                cur_start = offset
                cur_len = 1
            else:
                cur_len += 1
        else:
            if cur_len >= 20:
                return cur_start
            cur_start = -1
            cur_len = 0
    if cur_len >= 20:
        return cur_start
    return -1


def _read_ptr(base: int, byte_offset: int) -> int:
    raw = bytes((ctypes.c_ubyte * 8).from_address(base + byte_offset))
    return struct.unpack_from('<Q', raw, 0)[0]


class MgbaCore:
    def __init__(self, rom_path: str, save_path: str | None = None):
        self._ptr = _lib.GBACoreCreate()
        if not self._ptr:
            raise RuntimeError("GBACoreCreate failed")

        self._closed = False
        self._fn_deinit = None

        try:
            self._vt_off = _find_vtable_offset(self._ptr)
            if self._vt_off < 0:
                raise RuntimeError("Could not locate vtable in mCore struct")

            self._fn_deinit = self._make_fn('deinit', ctypes.CFUNCTYPE(None, ctypes.c_void_p))

            init_fn = self._make_fn('init', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p))
            if not init_fn(self._ptr):
                raise RuntimeError("Core init failed")

            _lib.mCoreInitConfig(self._ptr, None)

            self._video_buffer = (ctypes.c_uint32 * (GBA_W * GBA_H))()
            set_vbuf = self._make_fn('setVideoBuffer',
                                     ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t))
            set_vbuf(self._ptr, self._video_buffer, GBA_W)

            if not _lib.mCoreLoadFile(self._ptr, rom_path.encode()):
                raise RuntimeError(f"Failed to load ROM: {rom_path}")

            if save_path and Path(save_path).exists():
                _lib.mCoreLoadSaveFile(self._ptr, save_path.encode(), 0)

            self._fn_reset = self._make_fn('reset', ctypes.CFUNCTYPE(None, ctypes.c_void_p))
            self._fn_run_frame = self._make_fn('runFrame', ctypes.CFUNCTYPE(None, ctypes.c_void_p))
            self._fn_set_keys = self._make_fn('setKeys', ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32))
            self._fn_frame_counter = self._make_fn('frameCounter', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p))

            self._fn_reset(self._ptr)

        except Exception:
            if self._fn_deinit and self._ptr:
                self._fn_deinit(self._ptr)
            self._ptr = None
            self._closed = True
            raise

    def _make_fn(self, name: str, fn_type):
        ptr = _read_ptr(self._ptr, self._vt_off + _VT[name] * 8)
        if ptr == 0:
            raise RuntimeError(f"Vtable entry '{name}' is null")
        return fn_type(ptr)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Core is closed")

    def reset(self) -> None:
        self._check_open()
        self._fn_reset(self._ptr)

    def run_frame(self) -> None:
        self._check_open()
        self._fn_run_frame(self._ptr)

    def set_keys(self, keys: int) -> None:
        self._check_open()
        self._fn_set_keys(self._ptr, keys & 0xFFFFFFFF)

    def get_screen(self) -> np.ndarray:
        self._check_open()
        raw = np.frombuffer(self._video_buffer, dtype=np.uint8).reshape(GBA_H, GBA_W, 4).copy()
        return raw[:, :, [2, 1, 0]]

    def get_screen_bgra(self) -> np.ndarray:
        self._check_open()
        return np.frombuffer(self._video_buffer, dtype=np.uint8).reshape(GBA_H, GBA_W, 4).copy()

    def load_state_slot(self, slot: int) -> bool:
        self._check_open()
        return _lib.mCoreLoadState(self._ptr, slot, SAVESTATE_ALL)

    def save_state_slot(self, slot: int) -> bool:
        self._check_open()
        return _lib.mCoreSaveState(self._ptr, slot, SAVESTATE_ALL)

    def load_state(self, path: str) -> bool:
        self._check_open()
        return _lib.mCoreLoadStateNamed(self._ptr, path.encode(), SAVESTATE_ALL)

    def save_state(self, path: str) -> bool:
        self._check_open()
        return _lib.mCoreSaveStateNamed(self._ptr, path.encode(), SAVESTATE_ALL)

    @property
    def frame_counter(self) -> int:
        self._check_open()
        return self._fn_frame_counter(self._ptr)

    def read8(self, address: int) -> int:
        self._check_open()
        if not hasattr(self, '_fn_bus_read8'):
            self._fn_bus_read8 = self._make_fn(
                'busRead8', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32))
        return self._fn_bus_read8(self._ptr, address) & 0xFF

    def read16(self, address: int) -> int:
        self._check_open()
        if not hasattr(self, '_fn_bus_read16'):
            self._fn_bus_read16 = self._make_fn(
                'busRead16', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32))
        return self._fn_bus_read16(self._ptr, address) & 0xFFFF

    def read32(self, address: int) -> int:
        self._check_open()
        if not hasattr(self, '_fn_bus_read32'):
            self._fn_bus_read32 = self._make_fn(
                'busRead32', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32))
        return self._fn_bus_read32(self._ptr, address)

    def close(self) -> None:
        if not self._closed and self._ptr:
            self._fn_deinit(self._ptr)
            self._ptr = None
            self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
