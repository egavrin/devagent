; Python tree-sitter query for symbol extraction
; Based on Aider's approach with definitions and references

; Module-level assignments (constants)
(module
  (expression_statement
    (assignment
      left: (identifier) @name.definition.constant)
    @definition.constant))

; Class definitions
(class_definition
  name: (identifier) @name.definition.class) @definition.class

; Function definitions
(function_definition
  name: (identifier) @name.definition.function) @definition.function

; Method definitions (functions inside classes)
(class_definition
  body: (block
    (function_definition
      name: (identifier) @name.definition.method) @definition.method))

; Async function definitions
(function_definition
  (async) @async
  name: (identifier) @name.definition.async_function) @definition.async_function

; Decorators
(decorated_definition
  (decorator) @decorator)

; Import statements
(import_statement) @import
(import_from_statement) @import

; Function and method calls (references)
(call
  function: [
    (identifier) @name.reference.call
    (attribute
      attribute: (identifier) @name.reference.call)
  ]) @reference.call

; Attribute access (references)
(attribute
  attribute: (identifier) @name.reference.attribute) @reference.attribute

; Variable references
(identifier) @name.reference.variable

; Type annotations
(annotation
  (identifier) @name.reference.type)

; Exception handling
(except_clause
  (as_pattern
    alias: (identifier) @name.definition.exception))

; With statement aliases
(with_statement
  (with_clause
    (with_item
      (as_pattern
        alias: (identifier) @name.definition.context))))

; List/dict comprehensions
(list_comprehension
  (for_in_clause
    left: (identifier) @name.definition.comprehension))
(dictionary_comprehension
  (for_in_clause
    left: (identifier) @name.definition.comprehension))

; Lambda expressions
(lambda
  parameters: (lambda_parameters
    (identifier) @name.definition.parameter))

; Global and nonlocal declarations
(global_statement
  (identifier) @name.reference.global)
(nonlocal_statement
  (identifier) @name.reference.nonlocal)