import Cocoa
import WebKit

// Prometheus Worlds: a window around the viewer page. On first launch it runs setup.sh (tools, Python environments,
// models) and shows its progress; then it starts the project's private local server (thread/server.py), shows the
// page, and stops the server on quit. The repository path is stored in Resources/root.txt when the app is built.
final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var server: Process?
    var setup: Process?
    var repo = ""
    var port = 0
    var tries = 0
    var pageReady = false
    var pending: [String] = []

    func freePort() -> Int {
        let s = socket(AF_INET, SOCK_STREAM, 0)
        var a = sockaddr_in(); a.sin_family = sa_family_t(AF_INET); a.sin_addr.s_addr = inet_addr("127.0.0.1"); a.sin_port = 0
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &a) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(s, $0, len) } }
        _ = withUnsafeMutablePointer(to: &a) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(s, $0, &len) } }
        close(s)
        return Int(UInt16(bigEndian: a.sin_port))
    }

    // everything the pipelines run is inside the project folder; only system tools (sips, git, xcrun, curl) come from PATH
    func environment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = env["PATH"] ?? "/usr/bin:/bin"
        return env
    }

    func applicationDidFinishLaunching(_ n: Notification) {
        let rootFile = Bundle.main.url(forResource: "root", withExtension: "txt")
        var root = rootFile.flatMap { try? String(contentsOf: $0, encoding: .utf8) }?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if root.hasSuffix("/thread") { root = String(root.dropLast(7)) }      // apps built by older build scripts stored thread/
        repo = root
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "Prometheus Worlds"; window.center()
        web = WKWebView(frame: window.contentView!.bounds); web.autoresizingMask = [.width, .height]; web.navigationDelegate = self; web.uiDelegate = self
        window.contentView!.addSubview(web); window.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true)
        guard FileManager.default.fileExists(atPath: repo + "/setup.sh") else {
            return message("Project folder not found", "The app looks for the repository at:<br><code>\(repo)</code><br>Build the app again from the repository: open <code>Prometheus Worlds.xcodeproj</code> and press Run.")
        }
        if run(["--check"]) == 0 { startServer() } else { runSetup() }
    }

    func run(_ args: [String]) -> Int32 {
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/zsh"); p.arguments = [repo + "/setup.sh"] + args
        p.currentDirectoryURL = URL(fileURLWithPath: repo); p.environment = environment()
        p.standardOutput = FileHandle.nullDevice; p.standardError = FileHandle.nullDevice
        do { try p.run() } catch { return -1 }
        p.waitUntilExit(); return p.terminationStatus
    }

    // ---- first-run setup, with its output shown live in the window ----
    static let setupPage = """
    <!doctype html><meta charset=utf-8><style>
    body{margin:0;background:#0a0a0b;color:#e9e6e1;font:15px/1.5 -apple-system,system-ui,sans-serif;padding:48px 56px}
    h1{font:400 44px/1 Impact,'Arial Narrow Bold',sans-serif;text-transform:uppercase;margin:0 0 8px}
    p{color:#8f8b85;margin:0 0 24px;max-width:70ch} b#now{color:#ff8a1e;font-weight:600}
    pre{background:#131315;border:1px solid #2a2a2e;padding:16px;height:calc(100vh - 290px);overflow:auto;font:12px/1.5 ui-monospace,Menlo,monospace;color:#b9b4ad;white-space:pre-wrap;margin:0}
    .bar{height:3px;background:#ff4a12;width:0;transition:width .4s;margin-bottom:18px}
    </style><h1>Setting up</h1>
    <p>First launch: installing tools and models. This takes a while and needs the internet once. After this, everything runs offline.</p>
    <div class=bar id=bar></div><p>Now: <b id=now>starting</b></p><pre id=log></pre>
    <script>let n=0;function add(t){const l=document.getElementById('log');for(const line of t.split('\\n')){if(!line)continue;
    if(line.startsWith('STEP: ')){document.getElementById('now').textContent=line.slice(6);n++;document.getElementById('bar').style.width=Math.min(100,n*11)+'%'}
    l.textContent+=line+'\\n'}l.scrollTop=l.scrollHeight}</script>
    """

    func runSetup() {
        web.loadHTMLString(AppDelegate.setupPage, baseURL: nil)
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/zsh"); p.arguments = [repo + "/setup.sh"]
        p.currentDirectoryURL = URL(fileURLWithPath: repo); p.environment = environment()
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let d = h.availableData
            guard !d.isEmpty, let s = String(data: d, encoding: .utf8) else { return }
            DispatchQueue.main.async { self?.show(s) }
        }
        p.terminationHandler = { [weak self] p in
            pipe.fileHandleForReading.readabilityHandler = nil
            DispatchQueue.main.async {
                guard let self = self else { return }
                if p.terminationStatus == 0 { self.startServer() }
                else { self.show("\nSetup stopped. Fix the error above, then quit and open the app again: it carries on from where it stopped.\n") }
            }
        }
        do { try p.run(); setup = p } catch { message("Could not start setup", "\(error)") }
    }

    func show(_ s: String) {
        guard let data = try? JSONSerialization.data(withJSONObject: [s]), let json = String(data: data, encoding: .utf8) else { return }
        let js = "add(\(json)[0])"
        if pageReady { web.evaluateJavaScript(js) } else { pending.append(js) }
    }

    func message(_ title: String, _ body: String) {
        web.loadHTMLString("<body style='background:#0a0a0b;color:#e9e6e1;font:16px/1.6 system-ui;padding:48px 56px'><h2>\(title)</h2><p>\(body)</p></body>", baseURL: nil)
    }

    // ---- the viewer ----
    func startServer() {
        pageReady = false; pending = []
        let thread = repo + "/thread"
        port = freePort()
        let p = Process(); p.executableURL = URL(fileURLWithPath: thread + "/.venv/bin/python")
        p.arguments = [thread + "/server.py", String(port)]; p.currentDirectoryURL = URL(fileURLWithPath: thread)
        p.environment = environment()
        p.standardOutput = FileHandle.nullDevice; p.standardError = FileHandle.nullDevice
        do { try p.run(); server = p } catch { return message("Could not start the local server", "\(error)") }
        load()
    }
    func load() { web.load(URLRequest(url: URL(string: "http://127.0.0.1:\(port)/viewer/index.html")!)) }
    func webView(_ w: WKWebView, didFinish nav: WKNavigation!) {
        pageReady = true; for js in pending { w.evaluateJavaScript(js) }; pending = []
    }
    // the server needs a moment to start: retry until the page answers
    func webView(_ w: WKWebView, didFailProvisionalNavigation nav: WKNavigation!, withError e: Error) {
        guard server != nil else { return }
        tries += 1
        if tries < 40 { DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { self.load() } }
    }
    // lets <input type="file"> open the normal macOS file picker
    func webView(_ w: WKWebView, runOpenPanelWith p: WKOpenPanelParameters, initiatedByFrame f: WKFrameInfo, completionHandler done: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = p.allowsMultipleSelection; panel.canChooseDirectories = false
        panel.beginSheetModal(for: window) { r in done(r == .OK ? panel.urls : nil) }
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ n: Notification) { setup?.terminate(); server?.terminate() }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
let menu = NSMenu(); let item = NSMenuItem(); menu.addItem(item); let sub = NSMenu()
sub.addItem(NSMenuItem(title: "Quit Prometheus Worlds", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")); item.submenu = sub; app.mainMenu = menu
app.run()
