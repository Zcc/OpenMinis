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
}
