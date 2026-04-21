export function hasHelpFlag(args: ReadonlyArray<string>): boolean {
  return args.includes("--help") || args.includes("-h");
}

export function writeStdout(message = ""): void {
  process.stdout.write(message + "\n");
}

export function writeStderr(message = ""): void {
  process.stderr.write(message + "\n");
}
