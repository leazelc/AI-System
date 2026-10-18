import SwiftUI
import Foundation
import Combine

@MainActor
final class AppController: ObservableObject {
    @Published var lmStatus = "正在检查... / Checking..."
    @Published var proxyStatus = "正在检查... / Checking..."
    @Published var message = "AI-System 本地 AI 记忆系统 / Local AI Memory System"
    @Published var isStarting = false

    private var proxyProcess: Process?
    private var refreshTimer: Timer?
    private var launchToken = UUID()

    private var baseURL: URL {
        let bundled = Bundle.main.resourceURL?
            .appendingPathComponent("AI-System", isDirectory: true)
        if let bundled,
           FileManager.default.fileExists(
               atPath: bundled.appendingPathComponent("memory_proxy.py").path
           ) {
            return bundled
        }
        return URL(fileURLWithPath: NSHomeDirectory() + "/AI-System")
    }
    private let lmURL = URL(string: "http://127.0.0.1:1234/v1/models")!
    private let healthURL = URL(string: "http://127.0.0.1:8767/health")!

    init() {
        checkStatus()

        // Launch the memory proxy automatically when the app opens.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.check(url: self.healthURL) { [weak self] running in
                guard let self else { return }
                if running {
                    self.proxyStatus = "运行中 / Running"
                    self.message = "记忆代理已运行 / Memory Proxy is already running"
                } else {
                    self.startProxy()
                }
            }
        }

        refreshTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) {
            [weak self] _ in
            Task { @MainActor in
                self?.checkStatus()
            }
        }
    }

    deinit {
        refreshTimer?.invalidate()
    }

    func checkStatus() {
        check(url: lmURL) { [weak self] ok in
            self?.lmStatus = ok ? "已连接 / Connected" : "未连接 / Disconnected"
        }

        check(url: healthURL) { [weak self] ok in
            self?.proxyStatus = ok ? "运行中 / Running" : "未运行 / Not Running"
        }
    }

    private func check(
        url: URL,
        completion: @escaping (Bool) -> Void
    ) {
        var request = URLRequest(url: url)
        request.timeoutInterval = 2

        URLSession.shared.dataTask(with: request) { _, response, error in
            let ok = error == nil &&
                (response as? HTTPURLResponse)?.statusCode == 200

            DispatchQueue.main.async {
                completion(ok)
            }
        }.resume()
    }

    func startProxy() {
        guard !isStarting else { return }

        isStarting = true
        message = "正在检查记忆代理... / Checking Memory Proxy..."

        check(url: healthURL) { [weak self] alreadyRunning in
            guard let self else { return }

            if alreadyRunning {
                self.proxyStatus = "运行中 / Running"
                self.message = "记忆代理已经运行，无需重复启动。 / Memory Proxy is already running."
                self.isStarting = false
                return
            }

            self.launchProxy()
        }
    }

    private func launchProxy() {
        let script = baseURL.appendingPathComponent("memory_proxy.py")

        guard FileManager.default.fileExists(atPath: script.path) else {
            message = "找不到 memory_proxy.py：\(script.path)"
            isStarting = false
            return
        }

        let pythonCandidates = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
            "/Library/Developer/CommandLineTools/usr/bin/python3"
        ]

        guard let pythonPath = pythonCandidates.first(where: {
            FileManager.default.isExecutableFile(atPath: $0)
        }) else {
            message = "未找到 Python 3，请先安装。 / Python 3 was not found. Please install it."
            isStarting = false
            return
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = ["-u", script.path]
        process.currentDirectoryURL = baseURL

        // Keep bundled code separate from writable user memory data.
        // Development builds continue using ~/AI-System by default.
        var processEnvironment = ProcessInfo.processInfo.environment
        if baseURL.path.contains("/Contents/Resources/AI-System") {
            let appSupport = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support/AI-System", isDirectory: true)
            let userMemory = appSupport.appendingPathComponent("memory", isDirectory: true)

            do {
                try FileManager.default.createDirectory(
                    at: userMemory.appendingPathComponent("index", isDirectory: true),
                    withIntermediateDirectories: true
                )
                try FileManager.default.createDirectory(
                    at: userMemory.appendingPathComponent("archive", isDirectory: true),
                    withIntermediateDirectories: true
                )
                try FileManager.default.createDirectory(
                    at: userMemory.appendingPathComponent("candidates", isDirectory: true),
                    withIntermediateDirectories: true
                )
                try FileManager.default.createDirectory(
                    at: userMemory.appendingPathComponent("logs", isDirectory: true),
                    withIntermediateDirectories: true
                )
                processEnvironment["AI_SYSTEM_HOME"] = baseURL.path
                processEnvironment["AI_SYSTEM_MEMORY_ROOT"] = userMemory.path
            } catch {
                message = "无法创建用户记忆目录：\\(error.localizedDescription) / Could not create user memory directory."
                isStarting = false
                return
            }
        }
        processEnvironment["AI_SYSTEM_PORT"] = "8767"
        process.environment = processEnvironment

        let logURL = URL(fileURLWithPath: "/tmp/ai-system-desktop.log")
        FileManager.default.createFile(
            atPath: logURL.path,
            contents: nil
        )

        do {
            let logHandle = try FileHandle(forWritingTo: logURL)
            try logHandle.truncate(atOffset: 0)
            process.standardOutput = logHandle
            process.standardError = logHandle
        } catch {
            message = "无法创建启动日志：\(error.localizedDescription)"
            isStarting = false
            return
        }

        launchToken = UUID()
        let token = launchToken

        process.terminationHandler = { [weak self] terminatedProcess in
            DispatchQueue.main.async {
                guard let self, self.launchToken == token else { return }

                self.proxyProcess = nil
                self.isStarting = false

                if terminatedProcess.terminationStatus != 0 {
                    self.message =
                        "代理已退出，请查看启动日志。 / Proxy exited; check the startup log."
                } else {
                    self.message = "记忆代理已停止。 / Memory Proxy stopped."
                }

                self.checkStatus()
            }
        }

        do {
            try process.run()
            proxyProcess = process
            message = "进程已启动，正在等待健康检查... / Process started; waiting for health check..."

            verifyStartup(attempt: 0, token: token)
        } catch {
            message = "启动失败：\(error.localizedDescription)"
            isStarting = false
        }
    }

    private func verifyStartup(attempt: Int, token: UUID) {
        guard launchToken == token else { return }

        check(url: healthURL) { [weak self] ok in
            guard let self, self.launchToken == token else { return }

            if ok {
                self.proxyStatus = "运行中 / Running"
                self.message = "AI-System 记忆代理已成功启动。 / AI-System Memory Proxy started successfully."
                self.isStarting = false
                return
            }

            if let process = self.proxyProcess, !process.isRunning {
                self.message =
                    "代理启动失败，请查看启动日志。 / Proxy startup failed; check the log."
                self.isStarting = false
                return
            }

            if attempt >= 10 {
                self.message =
                    "等待代理超时，请查看启动日志。 / Timed out waiting for proxy; check the log."
                self.isStarting = false
                return
            }

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.verifyStartup(attempt: attempt + 1, token: token)
            }
        }
    }

    func stopProxy() {
        guard let process = proxyProcess, process.isRunning else {
            message = "代理不是由本应用启动的，不会擅自终止它。 / This app will not stop a proxy it did not start."
            checkStatus()
            return
        }

        launchToken = UUID()
        process.terminate()
        proxyProcess = nil
        isStarting = false
        message = "已请求停止本应用启动的代理。 / Requested to stop the proxy started by this app."
        checkStatus()
    }
}

struct StatusRow: View {
    let title: String
    let status: String

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Text(status)
                .foregroundStyle(
                    status == "已连接 / Connected" || status == "运行中 / Running"
                    ? .green : .secondary
                )
        }
        .padding(.vertical, 5)
    }
}

struct ContentView: View {
    @StateObject private var controller = AppController()

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 42))
                .foregroundStyle(.blue)

            Text("AI-System")
                .font(.system(size: 27, weight: .bold))

            Text("本地优先 · 个人 AI 记忆")
                .foregroundStyle(.secondary)

            Divider()

            StatusRow(title: "模型服务 / Model Service (LM Studio)", status: controller.lmStatus)
            StatusRow(title: "记忆代理 / Memory Proxy", status: controller.proxyStatus)

            Divider()

            Button("检查服务状态") {
                controller.checkStatus()
            }
            .buttonStyle(.bordered)

            HStack {
                Button("启动记忆代理 / Start Proxy") {
                    controller.startProxy()
                }
                .buttonStyle(.borderedProminent)
                .disabled(controller.isStarting)

                Button("停止记忆代理 / Stop Proxy") {
                    controller.stopProxy()
                }
                .buttonStyle(.bordered)
            }

            Text(controller.message)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)
        }
        .padding(28)
        .frame(width: 520, height: 420)
        .onAppear {
            controller.checkStatus()
        }
    }
}

@main
struct AISystemDesktopApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowResizability(.contentSize)
    }
}
