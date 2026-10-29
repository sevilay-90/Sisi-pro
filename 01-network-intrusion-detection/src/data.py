"""Loading and preparing the NSL-KDD network intrusion dataset."""

from pathlib import Path
import urllib.request

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]

# Every attack name in NSL-KDD mapped to one of four families.
ATTACK_FAMILY = {
    # Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "mailbomb": "DoS",
    # Probing / reconnaissance
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # Remote to Local
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L", "worm": "R2L",
    # User to Root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}


def download(force: bool = False) -> None:
    """Download the train and test files into data/ if missing."""
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES.values():
        target = DATA_DIR / name
        if force or not target.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, target)


def load(split: str) -> pd.DataFrame:
    """Return the requested split with `is_attack` and `family` columns added."""
    download()
    df = pd.read_csv(DATA_DIR / FILES[split], header=None, names=COLUMNS)
    df["family"] = df["label"].map(lambda x: "Normal" if x == "normal" else ATTACK_FAMILY.get(x, "Other"))
    df["is_attack"] = (df["label"] != "normal").astype(int)
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature columns only (labels and difficulty score removed)."""
    return df.drop(columns=["label", "difficulty", "family", "is_attack"])
