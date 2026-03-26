---
name: copilot-instructions-blueprint-generator
description: 'Technology-agnostic blueprint generator for creating comprehensive copilot-instructions.md files that guide GitHub Copilot to produce code consistent with project standards, architecture patterns, and exact technology versions by analyzing existing codebase patterns and avoiding assumptions.'
---

# Copilot Instructions Blueprint Generator

## Configuration Variables
${PROJECT_TYPE="Auto-detect|.NET|Java|JavaScript|TypeScript|React|Angular|Python|Multiple|Other"}
${ARCHITECTURE_STYLE="Layered|Microservices|Monolithic|Domain-Driven|Event-Driven|Serverless|Mixed"}
${CODE_QUALITY_FOCUS="Maintainability|Performance|Security|Accessibility|Testability|All"}
${DOCUMENTATION_LEVEL="Minimal|Standard|Comprehensive"}
${TESTING_REQUIREMENTS="Unit|Integration|E2E|TDD|BDD|All"}
${VERSIONING="Semantic|CalVer|Custom"}

## Generated Prompt

"Generate a comprehensive copilot-instructions.md file that will guide GitHub Copilot to produce code consistent with our project's standards, architecture, and technology versions. The instructions must be strictly based on actual code patterns in our codebase and avoid making any assumptions. Follow this approach:

### 1. Core Instruction Structure

```markdown
# GitHub Copilot Instructions

## Priority Guidelines

When generating code for this repository:

1. **Version Compatibility**: Always detect and respect the exact versions of languages, frameworks, and libraries used in this project
2. **Context Files**: Prioritize patterns and standards defined in the .github/copilot directory
3. **Codebase Patterns**: When context files don't provide specific guidance, scan the codebase for established patterns
4. **Architectural Consistency**: Maintain our architectural style and established boundaries
5. **Code Quality**: Prioritize the configured quality focus in all generated code
```

### 2. Codebase Analysis Instructions

To create the copilot-instructions.md file, first analyze the codebase to:

1. **Identify Exact Technology Versions**: Detect all programming languages, frameworks, and libraries by scanning file extensions and configuration files
2. **Understand Architecture**: Analyze folder structure and module organization
3. **Document Code Patterns**: Catalog naming conventions, documentation styles, error handling patterns
4. **Note Quality Standards**: Identify performance optimization techniques, security practices, testing approaches

### 3. Implementation Notes

The final copilot-instructions.md should:
- Be placed in the `.github/` directory
- Reference only patterns and standards that exist in the codebase
- Include explicit version compatibility requirements
- Avoid prescribing any practices not evident in the code
- Provide concrete examples from the codebase
- Be comprehensive yet concise enough for Copilot to effectively use

Important: Only include guidance based on patterns actually observed in the codebase. Explicitly instruct Copilot to prioritize consistency with existing code over external best practices or newer language features.
