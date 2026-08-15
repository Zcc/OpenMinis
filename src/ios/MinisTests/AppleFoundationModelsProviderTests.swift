import XCTest

final class AppleFoundationModelsProviderTests: XCTestCase {
    func testProviderMetadataIsCredentialFreeAndTextOnly() {
        let type = ProviderType.appleFoundationModels

        XCTAssertEqual(type.displayName, "Apple Foundation Models")
        XCTAssertEqual(type.builtInModels, [.appleSystemLanguageModel])
        XCTAssertEqual(type.defaultModality, .textOnly)
        XCTAssertTrue(LLMModel.appleSystemLanguageModel.capabilities.supportedAuth.isEmpty)
    }

    func testProviderTypeRoundTrips() throws {
        let data = try JSONEncoder().encode(ProviderType.appleFoundationModels)
        let decoded = try JSONDecoder().decode(ProviderType.self, from: data)

        XCTAssertEqual(decoded, .appleFoundationModels)
    }

    func testLocalInstanceDoesNotRequireKeychainCredential() {
        let instance = ProviderInstance(
            label: "On Device",
            providerType: .appleFoundationModels,
            credentialType: .apiKey
        )

        XCTAssertTrue(instance.hasAnyCredential)
    }

    func testPromptPreservesConversationRoles() {
        let messages = [
            LLMMessage(role: .user, content: "Hello"),
            LLMMessage(role: .assistant, content: "Hi"),
            LLMMessage(role: .user, content: "Continue"),
        ]

        XCTAssertEqual(
            AppleFoundationModelsProvider.prompt(from: messages),
            "User: Hello\n\nAssistant: Hi\n\nUser: Continue"
        )
    }

    func testAgentPromptIncludesToolHistory() {
        let messages = [
            AgentMessage(
                role: .assistant,
                parts: [.toolUse(id: "call-1", name: "lookup", input: ["query": "Minis"])]
            ),
            AgentMessage(
                role: .user,
                parts: [.toolResult(
                    id: "call-1",
                    name: "lookup",
                    content: "Found",
                    isError: false
                )]
            ),
        ]

        let prompt = AppleFoundationModelsProvider.prompt(from: messages)
        XCTAssertTrue(prompt.contains("[Tool call call-1: lookup"))
        XCTAssertTrue(prompt.contains("[Tool result call-1 from lookup: Found]"))
    }

    func testContextTruncationKeepsNewestContent() {
        let value = String(repeating: "old", count: 100) + "LATEST"
        let result = AppleFoundationModelsProvider.truncatedTail(value, limit: 12)

        XCTAssertTrue(result.hasPrefix("[Earlier conversation omitted"))
        XCTAssertTrue(result.hasSuffix("LATEST"))
        XCTAssertLessThan(result.count, value.count)
    }

    func testAgentInstructionsKeepSystemPromptEdgesWithinBudget() {
        let systemPrompt = "IDENTITY-" + String(repeating: "x", count: 2_000) + "-MEMORY"
        let result = AppleFoundationModelsProvider.agentInstructions(from: systemPrompt)

        XCTAssertTrue(result.contains("IDENTITY-"))
        XCTAssertTrue(result.hasSuffix("-MEMORY"))
        XCTAssertLessThan(result.count, 1_200)
    }

#if canImport(FoundationModels)
    func testSchemaIdentifiersAreGrammarSafe() {
        XCTAssertEqual(
            AppleFoundationModelsProvider.schemaIdentifier("browser_use.action-name"),
            "browser_use_action_name"
        )
        XCTAssertEqual(AppleFoundationModelsProvider.schemaIdentifier("123"), "_123")
    }
#endif
}
