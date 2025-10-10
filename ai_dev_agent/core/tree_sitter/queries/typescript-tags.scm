; TypeScript tree-sitter query for symbol extraction
; Includes type-specific constructs

; Interface declarations
(interface_declaration
  name: (type_identifier) @name.definition.interface) @definition.interface

; Type alias declarations
(type_alias_declaration
  name: (type_identifier) @name.definition.type) @definition.type

; Enum declarations
(enum_declaration
  name: (identifier) @name.definition.enum) @definition.enum

; Class declarations
(class_declaration
  name: (type_identifier) @name.definition.class) @definition.class

; Abstract classes
(class_declaration
  (abstract)
  name: (type_identifier) @name.definition.abstract_class) @definition.abstract_class

; Function declarations
(function_declaration
  name: (identifier) @name.definition.function) @definition.function

; Function signatures (in interfaces/types)
(function_signature
  name: (identifier) @name.definition.function_signature) @definition.function_signature

; Method signatures
(method_signature
  name: (property_identifier) @name.definition.method_signature) @definition.method_signature

; Property signatures
(property_signature
  name: (property_identifier) @name.definition.property_signature) @definition.property_signature

; Generic type parameters
(type_parameters
  (type_parameter
    (type_identifier) @name.definition.type_parameter))

; Namespace declarations
(module
  name: (identifier) @name.definition.namespace) @definition.namespace

; Import statements
(import_statement) @import
(import_specifier
  name: (identifier) @name.reference.import)

; Type imports
(import_statement
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @name.reference.type_import))))

; Export statements
(export_statement) @export

; Type exports
(export_statement
  (export_clause
    (export_specifier
      name: (identifier) @name.reference.type_export)))

; Function calls (references)
(call_expression
  function: [
    (identifier) @name.reference.call
    (member_expression
      property: (property_identifier) @name.reference.call)
  ]) @reference.call

; Type references
(type_identifier) @name.reference.type

; Type assertions
(as_expression
  (type_identifier) @name.reference.type_assertion)

; Type predicates
(predicate_type
  (type_identifier) @name.reference.type_predicate)

; Decorators
(decorator
  (identifier) @name.reference.decorator)

; Index signatures
(index_signature
  (identifier) @name.definition.index_parameter)

; Ambient declarations
(ambient_declaration) @ambient

; Const assertions
(as_expression
  (template_string) @const_assertion
  (#eq? @const_assertion "const"))