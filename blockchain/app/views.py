import datetime
import json
from urllib import response
from webbrowser import get
import sys
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from device_detector import DeviceDetector
from user_agents import parse

from flask import render_template, redirect, request, make_response

from app import app

CONNECTED_NODE_ADDRESS = "http://127.0.0.1:8000"

posts = []

def fetch_posts():
    get_chain_address = "{}/chain".format(CONNECTED_NODE_ADDRESS)
    response = requests.get(get_chain_address)
    print(response)
    if response.status_code == 200:
        content = []
        chain = json.loads(response.content)
        for block in chain["chain"]:
            for data in block["transactions"]:
                data["index"] = block["index"]
                data["hash"] = block["hash"]
                content.append(data)
            
        global posts
        posts=sorted(content, key=lambda k:k["timestamp"], reverse=True)

@app.route('/')
def index():
    fetch_posts()
    ua = request.headers.get('User-Agent')
    """page=requests.get("https://deviceinfo.me", verify=False, timeout=3000)
    soup = BeautifulSoup(page.content, 'html.parser')
    f=open("out.html", "x")
    f.write(str(soup))
    f.close()"""
    print(str(ua), file=sys.stderr)
    user_agent = parse(ua)
    print("OS: " + str(user_agent.os.family), file=sys.stderr)
    print("Version: " + str(user_agent.os.version), file=sys.stderr)
    print("Device: " + str(user_agent.device.family), file=sys.stderr)
    print("Brand: " + str(user_agent.device.brand), file=sys.stderr)
    print("Mobile: " + str(user_agent.is_mobile), file=sys.stderr)
    print("Tablet: " + str(user_agent.is_tablet), file=sys.stderr)
    print("Pc: " + str(user_agent.is_pc), file=sys.stderr)
    res = make_response(render_template('index.html', title='Covid-19 Case Confirmation',posts=posts,node_address=CONNECTED_NODE_ADDRESS,readable_time=timestamp_to_string))
    res.headers.set("Accept-Ch", "Device-Memory,Downlink,DPR,ECT,RTT,Save-Data,Sec-CH-Device-Memory,Sec-CH-Downlink,Sec-CH-DPR,Sec-CH-ECT,Sec-CH-Forced-Colors,Sec-CH-Prefers-Color-Scheme,Sec-CH-Prefers-Contrast,Sec-CH-Prefers-Reduced-Data,Sec-CH-Prefers-Reduced-Motion,Sec-CH-Prefers-Reduced-Transparency,Sec-CH-RTT,Sec-CH-Save-Data,Sec-CH-UA,Sec-CH-UA-Arch,Sec-CH-UA-Bitness,Sec-CH-UA-Full-Version,Sec-CH-UA-Full-Version-List,Sec-CH-UA-Mobile,Sec-CH-UA-Model,Sec-CH-UA-Platform,Sec-CH-UA-Platform-Version,Sec-CH-UA-WoW64,Sec-CH-Viewport-Height,Sec-CH-Viewport-Width,Sec-CH-Width,Viewport-Height,Viewport-Width,Width")
    return res
    
@app.route('/submit', methods=['POST'])
def submit_textarea():
    post_content = request.form["content"]
    author = request.form["author"]
    post_object = {
        'author': author,
        'content': post_content,
    }
    new_tx_address = "{}/new_transaction".format(CONNECTED_NODE_ADDRESS)
    requests.post(new_tx_address,
                  json=post_object,
                  headers={'Content-type': 'application/json'})
    return redirect('/')


def timestamp_to_string(epoch_time):
    return datetime.datetime.fromtimestamp(epoch_time).strftime('%H:%M')