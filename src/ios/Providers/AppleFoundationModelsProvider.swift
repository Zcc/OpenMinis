import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

enum AppleFoundationModelsError: LocalizedError {
    case unavailable(String)
    case unsupportedMedia

    var errorDescription: String? {
        switch self {
        case .unavailable(let reason):
            return String(localized: "Apple Foundation Models is unavailable: \(reason)")
        case .unsupportedMedia:
            return String(localized: "Apple Foundation Models currently supports text input only.")
        }
    }
}

final class AppleFoundationModelsProvider: LLMProvider, AgentProvider {
    let name = "Apple Foundation Models"
    var model: LLMModel
    var defaultMaxTokens: Int { model.maxOutputTokens ?? 512 }

    init(model: LLMModel = .appleSystemLanguageModel) {
        self.model = model
    }

    func sendMessage(
        messages: [LLMMessage],
        systemPrompt: String?,
        maxTokens: Int,
        temperature: Double?
    ) async throws -> LLMResponse {
        try validate(messages: messages)
#if canImport(FoundationModels)
        guard #available(iOS 26.0, *) else {
            throw AppleFoundationModelsError.unavailable("iOS 26 or later is required")
        }
        try Self.checkAvailability()
        let session = LanguageModelSession(
            model: .default,
            tools: [],
            instructions: compact(systemPrompt, limit: 800)
        )
        let response = try await session.respond(
            to: Self.truncatedTail(Self.prompt(from: messages), limit: 2_200),
            options: Self.options(maxTokens: maxTokens, temperature: temperature)
        )
        return LLMResponse(text: response.content, stopReason: "end_turn", usage: nil)
#else
        throw AppleFoundationModelsError.unavailable("this build does not include the FoundationModels framework")
#endif
    }

    func streamMessage(
        messages: [LLMMessage],
        systemPrompt: String?,
        maxTokens: Int,
        temperature: Double?
    ) async throws -> AsyncThrowingStream<LLMStreamChunk, Error> {
        try validate(messages: messages)
#if canImport(FoundationModels)
        guard #available(iOS 26.0, *) else {
            throw AppleFoundationModelsError.unavailable("iOS 26 or later is required")
        }
        try Self.checkAvailability()
        let session = LanguageModelSession(
            model: .default,
            tools: [],
            instructions: compact(systemPrompt, limit: 800)
        )
        let snapshots = session.streamResponse(
            to: Self.truncatedTail(Self.prompt(from: messages), limit: 2_200),
            options: Self.options(maxTokens: maxTokens, temperature: temperature)
        )
        return AsyncThrowingStream { continuation in
            let task = Task {
                continuation.yield(.started)
                var previous = ""
                do {
                    for try await snapshot in snapshots {
                        try Task.checkCancellation()
                        let current = snapshot.content
                        let delta = current.hasPrefix(previous)
                            ? String(current.dropFirst(previous.count))
                            : current
                        if !delta.isEmpty {
                            continuation.yield(.text(delta))
                        }
                        previous = current
                    }
                    continuation.yield(.finished(stopReason: "end_turn"))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
#else
        throw AppleFoundationModelsError.unavailable("this build does not include the FoundationModels framework")
#endif
    }

    func streamAgentMessageClamped(
        messages: [AgentMessage],
        systemPrompt: String?,
        tools: [AgentToolDefinition],
        maxTokens: Int,
        thinkingLevel: ThinkingLevel
    ) async throws -> AsyncThrowingStream<AgentStreamEvent, Error> {
        try validate(messages: messages)
#if canImport(FoundationModels)
        guard #available(iOS 26.0, *) else {
            throw AppleFoundationModelsError.unavailable("iOS 26 or later is required")
        }
        try Self.checkAvailability()
        let adapters = try tools.map(Self.makeTool)
        let session = LanguageModelSession(
            model: .default,
            tools: adapters,
            instructions: Self.agentInstructions(from: systemPrompt)
        )
        let snapshots = session.streamResponse(
            to: Self.truncatedTail(Self.prompt(from: messages), limit: 1_000),
            options: Self.options(maxTokens: maxTokens, temperature: nil)
        )
        return AsyncThrowingStream { continuation in
            let task = Task {
                var previous = ""
                var startedText = false
                do {
                    for try await snapshot in snapshots {
                        try Task.checkCancellation()
                        let current = snapshot.content
                        let delta = current.hasPrefix(previous)
                            ? String(current.dropFirst(previous.count))
                            : current
                        if !delta.isEmpty {
                            if !startedText {
                                continuation.yield(.contentBlockStart(.text))
                                startedText = true
                            }
                            continuation.yield(.textDelta(delta))
                        }
                        previous = current
                    }
                    continuation.yield(.done(stopReason: .endTurn))
                    continuation.finish()
                } catch let error as LanguageModelSession.ToolCallError {
                    if let invocation = error.underlyingError as? ToolInvocation {
                        let id = "apple-\(UUID().uuidString)"
                        let args = Self.decodeArguments(invocation.jsonArguments)
                        continuation.yield(.contentBlockStart(.toolUse(id: id, name: invocation.toolName)))
                        continuation.yield(.toolCallComplete(id: id, name: invocation.toolName, args: args, metadata: nil))
                        continuation.yield(.done(stopReason: .toolUse))
                        continuation.finish()
                    } else {
                        continuation.finish(throwing: error)
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
#else
        throw AppleFoundationModelsError.unavailable("this build does not include the FoundationModels framework")
#endif
    }

    private func validate(messages: [LLMMessage]) throws {
        guard messages.allSatisfy({ $0.images.isEmpty && $0.audios.isEmpty }) else {
            throw AppleFoundationModelsError.unsupportedMedia
        }
    }

    private func validate(messages: [AgentMessage]) throws {
        for message in messages {
            for part in message.parts {
                if case .imageData = part {
                    throw AppleFoundationModelsError.unsupportedMedia
                }
                if case .toolResult(_, _, _, _, let imageData, _, _, _) = part, imageData != nil {
                    throw AppleFoundationModelsError.unsupportedMedia
                }
            }
        }
    }

    private func compact(_ value: String?, limit: Int) -> String? {
        guard let value, !value.isEmpty else { return nil }
        return String(value.prefix(limit))
    }

    static func prompt(from messages: [LLMMessage]) -> String {
        messages.map { "\($0.role == .user ? "User" : "Assistant"): \($0.content)" }
            .joined(separator: "\n\n")
    }

    static func prompt(from messages: [AgentMessage]) -> String {
        messages.map { message in
            let body = message.parts.map { part -> String in
                switch part {
                case .text(let text):
                    return text
                case .toolUse(let id, let name, let input):
                    return "[Tool call \(id): \(name) \(jsonString(input))]"
                case .toolResult(let id, let name, let content, let isError, _, _, _, _):
                    return "[Tool result \(id) from \(name)\(isError ? " (error)" : ""): \(content)]"
                case .imageData:
                    return "[Unsupported image]"
                }
            }.joined(separator: "\n")
            return "\(message.role == .user ? "User" : "Assistant"): \(body)"
        }.joined(separator: "\n\n")
    }

    static func jsonString(_ value: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
              let string = String(data: data, encoding: .utf8) else { return "{}" }
        return string
    }

    static func truncatedTail(_ value: String, limit: Int) -> String {
        guard value.count > limit else { return value }
        return "[Earlier conversation omitted to fit the on-device context window]\n\n"
            + value.suffix(limit)
    }

    static func agentInstructions(from systemPrompt: String?) -> String {
        guard let systemPrompt, !systemPrompt.isEmpty else { return compactAgentInstructions }
        let excerpt: String
        if systemPrompt.count <= 600 {
            excerpt = systemPrompt
        } else {
            excerpt = String(systemPrompt.prefix(250))
                + "\n[Long system context omitted for the on-device model]\n"
                + systemPrompt.suffix(250)
        }
        return compactAgentInstructions + "\n\nRelevant Minis instructions:\n" + excerpt
    }

    private static let compactAgentInstructions = """
        You are Minis, a concise on-device assistant. Complete the user's task using the registered \
        tools when useful. Invoke tools through function calling, never by printing tool syntax. \
        Treat tool results as conversation context, report errors honestly, and do not claim an \
        action succeeded unless its tool result confirms success. Apple Foundation Models is \
        text-only, so explain when media cannot be processed directly.
        """

#if canImport(FoundationModels)
    @available(iOS 26.0, *)
    private struct ToolInvocation: Error {
        let toolName: String
        let jsonArguments: String
    }

    @available(iOS 26.0, *)
    private struct DynamicTool: FoundationModels.Tool {
        typealias Output = String
        typealias Arguments = GeneratedContent

        let name: String
        let description: String
        let parameters: GenerationSchema
        var includesSchemaInInstructions: Bool { true }

        func call(arguments: GeneratedContent) async throws -> String {
            throw ToolInvocation(toolName: name, jsonArguments: arguments.jsonString)
        }
    }

    @available(iOS 26.0, *)
    private static func checkAvailability() throws {
        switch SystemLanguageModel.default.availability {
        case .available:
            return
        case .unavailable(let reason):
            throw AppleFoundationModelsError.unavailable(String(describing: reason))
        @unknown default:
            throw AppleFoundationModelsError.unavailable("unknown availability state")
        }
    }

    @available(iOS 26.0, *)
    private static func options(maxTokens: Int, temperature: Double?) -> GenerationOptions {
        GenerationOptions(
            sampling: nil,
            temperature: temperature,
            maximumResponseTokens: max(1, min(maxTokens, 512))
        )
    }

    @available(iOS 26.0, *)
    private static func makeTool(_ tool: AgentToolDefinition) throws -> any FoundationModels.Tool {
        let properties = tool.parameters.map { name, parameter in
            DynamicGenerationSchema.Property(
                name: name,
                description: compactDescription(parameter.description, limit: 40),
                schema: dynamicSchema(
                    name: schemaIdentifier("\(tool.name)_\(name)"),
                    parameter: parameter
                ),
                isOptional: !tool.required.contains(name)
            )
        }
        let root = DynamicGenerationSchema(
            name: schemaIdentifier(tool.name),
            description: compactDescription(tool.description, limit: 60),
            properties: properties
        )
        let schema = try GenerationSchema(root: root, dependencies: [])
        return DynamicTool(
            name: tool.name,
            description: compactDescription(tool.description, limit: 60),
            parameters: schema
        )
    }

    @available(iOS 26.0, *)
    private static func dynamicSchema(name: String, parameter: AgentToolParam) -> DynamicGenerationSchema {
        if let values = parameter.enumValues, !values.isEmpty {
            return DynamicGenerationSchema(
                name: name,
                description: compactDescription(parameter.description, limit: 40),
                anyOf: values
            )
        }
        switch parameter.type {
        case .string:
            return DynamicGenerationSchema(type: String.self)
        case .integer:
            return DynamicGenerationSchema(type: Int.self)
        case .boolean:
            return DynamicGenerationSchema(type: Bool.self)
        }
    }

    private static func decodeArguments(_ json: String) -> [String: Any] {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        return object
    }

    private static func compactDescription(_ value: String, limit: Int) -> String {
        let firstSentence = value.split(separator: ".", maxSplits: 1).first.map(String.init) ?? value
        return String(firstSentence.prefix(limit))
    }

    static func schemaIdentifier(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_"))
        let sanitized = String(value.unicodeScalars.map {
            allowed.contains($0) ? Character($0) : "_"
        })
        guard let first = sanitized.unicodeScalars.first,
              CharacterSet.letters.union(CharacterSet(charactersIn: "_")).contains(first) else {
            return "_\(sanitized)"
        }
        return sanitized
    }
#endif
}
