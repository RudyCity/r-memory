# Agent Coding Rules & Standards

These rules apply to all code generation, refactoring, and additions in the `r-memory` project. Always adhere to these constraints without exception:

## 1. Clean Coding & Best Practices
*   **TypeScript-First**: Write strong, explicit types. Avoid using `any` unless absolutely necessary (e.g. dynamic CJS imports or untyped API returns). Use generics where appropriate.
*   **Error Handling**: Always write defensive code. Wrap async calls, file system, database, and network operations in proper `try-catch` blocks and throw informative, custom errors.
*   **Resource Management**: Always ensure database connections, file streams, and model sessions are properly closed or disposed of when no longer needed (e.g., using `finally` blocks).

## 2. Modular Architecture
*   **Single Responsibility Principle**: Each class, module, and function must have a single, well-defined responsibility (e.g., separate database adapters from embedding generation and document parsing utilities).
*   **Loose Coupling**: Design modules to communicate via clear interfaces (`types.ts`). Avoid direct dependencies between unrelated modules to make it easy to swap database drivers or embedding providers.
*   **Dynamic Loading**: Lazy-load heavy dependencies (like `@huggingface/transformers` or parser engines) to keep initial startup times fast.

## 3. Maintainability & Scalability
*   **Documented Interfaces**: Document all public-facing APIs, methods, and classes using standard JSDoc comments explaining parameters, return values, and thrown exceptions.
*   **Self-Documenting Code**: Choose clean, descriptive names for variables, classes, and methods. Avoid obscure abbreviations.
*   **Performance Optimization**: Use SQLite features like WAL journal mode, prepared statements, and transaction groupings (`db.transaction()`) for high-volume operations.

## 4. File Length Limit (Max 1000 Lines)
*   **Strict Limit**: No source code file (`.ts`, `.js`, etc.) may exceed **1,000 lines of code**.
*   **Refactoring Constraint**: If a file grows near or beyond 1,000 lines, it MUST be broken down into smaller sub-modules or utilities (e.g. moving helper functions to a separate file in `utils/`).
