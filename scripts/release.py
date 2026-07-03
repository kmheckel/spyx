#!/usr/bin/env python3
import argparse
import subprocess
import sys

import tomllib


def run_command(command, check=True):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result

def get_version():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]

def main():
    parser = argparse.ArgumentParser(description="Release utility for spyx")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--github", action="store_true", help="Create GitHub release")
    parser.add_argument("--pypi", action="store_true", help="Publish to PyPI")
    
    args = parser.parse_args()

    version = get_version()
    tag = f"v{version}"
    
    print(f"Releasing version {version}...")

    if args.dry_run:
        print("DRY RUN: Commands will not be executed.")

    # 1. Build the package
    build_cmd = ["uv", "build"]
    if args.dry_run:
        print(f"Would run: {' '.join(build_cmd)}")
    else:
        run_command(build_cmd)

    # 2. Tag and Release on GitHub
    if args.github:
        tag_cmd = ["git", "tag", "-a", tag, "-m", f"Release {tag}"]
        push_tag_cmd = ["git", "push", "origin", tag]
        gh_release_cmd = ["gh", "release", "create", tag, "--title", tag, "--notes", f"Release {tag}"]
        
        if args.dry_run:
            print(f"Would run: {' '.join(tag_cmd)}")
            print(f"Would run: {' '.join(push_tag_cmd)}")
            print(f"Would run: {' '.join(gh_release_cmd)}")
        else:
            try:
                run_command(tag_cmd)
                run_command(push_tag_cmd)
                run_command(gh_release_cmd)
            except subprocess.CalledProcessError as e:
                print(f"Error during GitHub release: {e}", file=sys.stderr)
                sys.exit(1)

    # 3. Publish to PyPI
    if args.pypi:
        publish_cmd = ["uv", "publish"]
        if args.dry_run:
            print(f"Would run: {' '.join(publish_cmd)}")
        else:
            try:
                run_command(publish_cmd)
            except subprocess.CalledProcessError as e:
                print(f"Error during PyPI publish: {e}", file=sys.stderr)
                sys.exit(1)

    print(f"Successfully processed release for {tag}")

if __name__ == "__main__":
    main()
