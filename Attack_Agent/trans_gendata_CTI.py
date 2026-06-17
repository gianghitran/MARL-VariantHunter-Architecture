# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : trans_gendata.py
# Description：visualize
"""
import os
import re
import argparse
import shutil


def list_files(directory):
    files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

def make_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return 1

def transdata_one_row(trans_file,generate_file,filename,max_node):
    file_path = trans_file + "\\" + filename
    new_file_path = generate_file + "\\" + "new_" + filename
    f = open(file_path, "r", encoding="utf-8")
    graph_lines = f.readlines()
    f.close()
    with open(new_file_path, 'w',encoding="utf-8") as file_new:
        for data in graph_lines:
            entities = []
            other_data = []
            for item in data.split(" "):
                if item.startswith("*"):
                    entity = item[1:]
                    entities.append(entity)
                else:
                    other_data.append(item)
            # new_data = " ".join(entities + other_data)
            new_data = ""
            if len(entities) == max_node:
                for entity in entities:
                    new_data = new_data + " " + entity
            else:
                for entity in entities[:-1]:
                    new_data = new_data + " " + entity
            for od in other_data:
                new_data = new_data + od + " "
            if new_data.endswith(" "):
                new_data = new_data[1:-1]
            new_data = new_data.replace("\n", "")
            file_new.write(new_data)
            file_new.write("\n")

def transdata_multi_row(generated_file,multirow_generated_file,filename):
    file_path = generated_file + "\\" + filename
    new_file_path = multirow_generated_file + "\\" + filename
    f = open(file_path, "r", encoding="utf-8")
    action_dic = {"RD":"RD","WR":"WR","EX":"EX","UK":"UK","CD":"CD","FR":"FR","IJ":"IJ","ST":"ST","RF":"RF"}
    graph_lines = f.readlines()
    f.close()
    i = 0
    entity_list = ["MP","TP","MF","SF","TF","SO"]
    node_len = []
    edge_len = []
    with open(new_file_path, 'w') as file:
        for line in graph_lines:
            file.write("#"+str(i)+"\n")
            i += 1
            split = line.split(" ")
            x1 = 0
            for item in split:
                if item in entity_list:
                    x1 += 1
            file.write(str(x1)+"\n")
            node_len.append(x1)
            for item in split:
                if item in entity_list:
                    file.write(item+"\n")
            split_new = []
            for i1 in range(x1,len(split)):
                split_new.append(split[i1])
            file.write(str(int(len(split_new)/3))+"\n")
            edge_len.append(int(len(split_new)/3))
            for item in range(0, len(split_new), 3):
                edge = split_new[item] + " " + split_new[item+1] + " "+action_dic[split_new[item+2].replace("\n","")]
                file.write(edge+"\n")
            file.write("\n")
    node_ave = sum(node_len)/len(node_len)
    edge_ave = sum(edge_len)/len(edge_len)
    print("node_ave")
    print(node_ave)
    print("edge_ave")
    print(edge_ave)
    return node_ave,edge_ave

def graph_txt_construct(multirow_path,trans_multirow_path):
    files_and_directories = os.listdir(multirow_path)
    files = [f for f in files_and_directories if os.path.isfile(os.path.join(multirow_path, f))]
    for file in files:
        file_path = multirow_path +"\\" + file
        f = open(file_path, "r")
        graph_lines = f.readlines()
        f.close()
        i = 1
        desktop_path = trans_multirow_path+"\\"
        if not os.path.exists(desktop_path):
            os.makedirs(desktop_path)
        else:
            shutil.rmtree(desktop_path)
            os.makedirs(desktop_path)

        for line in graph_lines:
            line = line.strip()
            if re.match("#[0-9]+", line):
                filename = desktop_path+str(i)+".txt"
                i += 1
                f = open(filename, "a+")
            elif line != "\n":
                f.write(line+"\n")
            else:
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphFlow model')
    parser.add_argument('--directory_path', type=str, default=r'.\save_pretrain\exp_ASG_CTI_epoch3_1gpu\asg', help='file to transfer')
    parser.add_argument('--new_directory_path', type=str,default=r'.\save_pretrain\exp_ASG_CTI_epoch3_1gpu\onerow_asg',help='onerow file to store')
    parser.add_argument('--new_directory_path_multirow', type=str,default=r'.\save_pretrain\exp_ASG_CTI_epoch3_1gpu\multirow_asg',help='multirow file to store')
    parser.add_argument('--new_directory_path_multirow_divide', type=str,default=r'.\save_pretrain\exp_ASG_CTI_epoch3_1gpu\multirow_asg_divide',help='divided multirow file to store')
    parser.add_argument('--maxnode', type=int, help='max node of the dataset', required=True)
    args = parser.parse_args()
    file_names = list_files(args.directory_path)
    make_directory(args.new_directory_path)
    make_directory(args.new_directory_path_multirow)
    make_directory(args.new_directory_path_multirow_divide)
    for file in file_names:
        if file.endswith(".txt"):
            transdata_one_row(args.directory_path,args.new_directory_path,file,args.maxnode)
    new_file_names = list_files(args.new_directory_path)
    for file in new_file_names:
        if file.endswith(".txt"):
            transdata_multi_row(args.new_directory_path,args.new_directory_path_multirow,file)
    graph_txt_construct(args.new_directory_path_multirow,args.new_directory_path_multirow_divide)


def convert_to_dot(raw_txt_path, out_dot_path, technique_hints=None):
    """
    Convert MaskGAF raw text output sang .dot format.

    Args:
        raw_txt_path: đường dẫn tới file text output từ generate_molecule
        out_dot_path: đường dẫn lưu file .dot
        technique_hints: dict từ mapping_rules (optional)
            {"allowed_processes": [...], "hub_process": "bash", ...}
    """
    import re
    import random

    if not os.path.exists(raw_txt_path):
        print(f"[convert_to_dot] File not found: {raw_txt_path}")
        return

    with open(raw_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print(f"[convert_to_dot] Empty file: {raw_txt_path}")
        return

    entity_list = ["MP", "TP", "MF", "SF", "TF", "SO"]
    action_dic = {"RD": "RD", "WR": "WR", "EX": "EX", "UK": "UK",
                  "CD": "CD", "FR": "FR", "IJ": "IJ", "ST": "ST", "RF": "RF"}

    # Parse the first line (single-line format from generate_molecule)
    line = lines[0].strip()
    tokens = line.split(" ")

    # Extract nodes and edges
    nodes = []
    edge_tokens = []
    for token in tokens:
        clean = token.lstrip("*")
        if clean in entity_list:
            nodes.append(clean)
        else:
            edge_tokens.append(token)

    if not nodes:
        print(f"[convert_to_dot] No nodes found in {raw_txt_path}")
        return

    # Parse edges: every 3 tokens = (src_idx, dst_idx, verb)
    edges = []
    for i in range(0, len(edge_tokens) - 2, 3):
        try:
            src = int(edge_tokens[i])
            dst = int(edge_tokens[i + 1])
            verb = edge_tokens[i + 2].split("-")[0]  # strip stage annotation
            if verb in action_dic:
                edges.append((src, dst, action_dic[verb]))
        except (ValueError, IndexError):
            continue

    # Generate instance names based on hints
    hub_proc = "bash"
    if technique_hints:
        hub_proc = technique_hints.get("hub_process", "bash")
        allowed_procs = technique_hints.get("allowed_processes", ["bash", "python3"])
    else:
        allowed_procs = ["bash", "python3", "sh", "curl", "wget"]

    file_names = ["data.log", "output.txt", "config.conf", "payload.bin",
                  "creds.txt", "scan.xml", "dump.tmp", "script.sh"]
    socket_names = ["0.0.0.0:80", "0.0.0.0:443", "0.0.0.0:22",
                    "0.0.0.0:4444", "0.0.0.0:8080"]

    # Build DOT file
    dot_lines = ["digraph {"]
    for idx, ntype in enumerate(nodes):
        if idx >= len(nodes):
            break
        if ntype in ("MP", "TP"):
            if idx == 0:
                label = hub_proc
            else:
                label = random.choice(allowed_procs)
            dot_lines.append(f'  n{idx} [label="{label}" type=process]')
        elif ntype in ("MF", "SF", "TF"):
            label = random.choice(file_names)
            dot_lines.append(f'  n{idx} [label="{label}" type=file]')
        elif ntype == "SO":
            label = random.choice(socket_names)
            dot_lines.append(f'  n{idx} [label="{label}" type=socket]')

    for src, dst, verb in edges:
        if src < len(nodes) and dst < len(nodes):
            dot_lines.append(f'  n{src} -> n{dst} [label="{verb}"]')

    dot_lines.append("}")

    with open(out_dot_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dot_lines))
    print(f"[convert_to_dot] Saved {len(nodes)} nodes, {len(edges)} edges -> {out_dot_path}")
