; JavaScript/TypeScript tree-sitter query for symbol extraction
; Supports both definitions and references

; Class declarations
(class_declaration
  name: (identifier) @name.definition.class) @definition.class

; Function declarations
(function_declaration
  name: (identifier) @name.definition.function) @definition.function

; Function expressions with names
(function_expression
  name: (identifier) @name.definition.function) @definition.function

; Arrow functions assigned to variables
(variable_declarator
  name: (identifier) @name.definition.function
  value: (arrow_function)) @definition.function

; Variable declarations
(variable_declarator
  name: (identifier) @name.definition.variable) @definition.variable

; Method definitions
(method_definition
  name: (property_identifier) @name.definition.method) @definition.method

; Property definitions in objects
(pair
  key: (property_identifier) @name.definition.property
  value: (_)) @definition.property

; Import statements
(import_statement) @import
(import_specifier
  name: (identifier) @name.reference.import)

; Export statements
(export_statement) @export

; Function calls (references)
(call_expression
  function: [
    (identifier) @name.reference.call
    (member_expression
      property: (property_identifier) @name.reference.call)
  ]) @reference.call

; Member access (references)
(member_expression
  property: (property_identifier) @name.reference.member) @reference.member

; Variable references
(identifier) @name.reference.variable

; JSX elements and components
(jsx_element
  open_tag: (jsx_opening_element
    name: (identifier) @name.reference.component))

; Object destructuring
(object_pattern
  (shorthand_property_identifier_pattern) @name.definition.destructured)
(object_pattern
  (pair_pattern
    key: (property_identifier)
    value: (identifier) @name.definition.destructured))

; Array destructuring
(array_pattern
  (identifier) @name.definition.destructured)

; Generator functions
(generator_function_declaration
  name: (identifier) @name.definition.generator) @definition.generator

; Async functions
(function_declaration
  (async) @async
  name: (identifier) @name.definition.async_function) @definition.async_function

; Constructor
(class_declaration
  body: (class_body
    (method_definition
      name: (property_identifier) @name.definition.constructor
      (#eq? @name.definition.constructor "constructor"))))

; Static methods
(class_declaration
  body: (class_body
    (method_definition
      (static)
      name: (property_identifier) @name.definition.static_method))))

; Getters and setters
(method_definition
  (get)
  name: (property_identifier) @name.definition.getter) @definition.getter

(method_definition
  (set)
  name: (property_identifier) @name.definition.setter) @definition.setter