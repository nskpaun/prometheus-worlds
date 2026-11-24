"""Blender with no window, serving the Blender MCP addon's commands (run by blender_world.py, inside Blender):
  blender -b --factory-startup --python blender_host.py -- ADDON.py PORT
The addon (ahujasid's "MCP for Blender", the copy bundled in its Python package) refuses to start its server without a
window, because Blender runs no timers then and its queue of commands would never be emptied. Here the same server is
started by hand and this script empties the queue itself, on Blender's main thread, until its parent goes away.
The addon's own timers (its check for updates among them) never run without a window, and none of its online asset or
generation services is switched on. blender_world.py also runs this process with the network closed to all but this Mac."""
import sys, os, time, socket, threading, importlib.util
import bpy

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

def serve(addon_path, port):
    kit = load("kit", os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_kit.py"))      # what the model's code imports
    kit._gpu()
    addon = load("blendermcp_addon", addon_path)
    addon.register()                                        # the scene settings its commands read; its panels are harmless without a window
    s = addon.BlenderMCPServer(host="127.0.0.1", port=port); bpy.types.blendermcp_server = s
    s.running = True                                        # what its start() does, less the timer
    s.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.socket.bind(("127.0.0.1", port)); s.socket.listen(5)
    s.server_thread = threading.Thread(target=s._server_loop, daemon=True); s.server_thread.start()
    print(f"HOST READY {port}", flush=True)
    parent = os.getppid()
    while s.running and os.getppid() == parent:             # the parent gone: nobody is left to ask anything
        s._drain_command_queue(); time.sleep(0.02)

if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:]; serve(a[0], int(a[1]))
