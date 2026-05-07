import subprocess
import sys
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=str, required=True, help="Le chemin vers le dossier contenant les data")
parser.add_argument("--use_mask", action="store_true")
parser.add_argument("-r", type=str, default="1", help="La liste des résolutions à lancer")
args = parser.parse_args()

script_a_lancer = "train.py"
liste_res = args.r.split(" ")

chemins_extraits = []

for res in liste_res:
    commande = [sys.executable, "-u", script_a_lancer, "-s", args.s, "-r", res]
    if args.use_mask:
        commande.extend(["--use_mask", "True", "--random_background"])

    print(f"\n--- Lancement résolution {res} ---")
    process = subprocess.Popen(
        commande, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )

    chemin_trouve = None
    for line in process.stdout:
        clean_line = line.strip('\r\n')
        if "Training progress" in clean_line:
            sys.stdout.write(f"\r{clean_line}")
            sys.stdout.flush()
        else:
            print(f"{clean_line}")
            
        match = re.search(r"Output folder:\s*(.*?)\s*\[", clean_line)
        if match:
            chemin_trouve = match.group(1).strip()
    process.wait()

    if process.returncode == 0:
        if chemin_trouve and os.path.exists(chemin_trouve):
            base_name = os.path.basename(args.s.strip(os.sep))
            suffix = f"_r{res}"
            if args.use_mask:
                suffix += "_mask"
            
            nouveau_nom = base_name + suffix
            nouveau_chemin = os.path.join(os.path.dirname(chemin_trouve), nouveau_nom)

            try:
                if os.path.exists(nouveau_chemin):
                    import shutil
                    shutil.rmtree(nouveau_chemin)
                
                os.rename(chemin_trouve, nouveau_chemin)
                chemins_extraits.append(nouveau_chemin)
                print(f"Dossier renommé en : {nouveau_nom}")
            except Exception as e:
                print(f"Erreur lors du renommage : {e}")
                chemins_extraits.append(chemin_trouve)
        else:
            print(f"{chemin_trouve} inexistant")
    else:
        print(process.returncode)

print("\nListe des dossier créés :")
for c in chemins_extraits:
    print(c)
