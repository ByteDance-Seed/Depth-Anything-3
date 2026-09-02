#!/usr/bin/env python3
# flake8: noqa: E501
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Depth Anything 3 Gallery Server (two-level, single-file)
Now supports paginated depth preview (4 per page).
"""

import argparse
import json
import mimetypes
import os
import posixpath
import sys
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote

# ------------------------------ Embedded HTML ------------------------------ #

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Depth Anything 3 Gallery</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="icon" href="https://i.postimg.cc/rFSzGJ7J/light-icon.jpg" media="(prefers-color-scheme: light)">
  <link rel="icon" href="https://i.postimg.cc/P5gZfJsf/dark-icon.jpg" media="(prefers-color-scheme: dark)">
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    :root {
      --gap:16px; --card-radius:16px; --shadow:0 8px 24px rgba(0,0,0,.12);
      --maxW:1036px; --maxH:518px;
      --tech-blue: #00d4ff;
      --tech-cyan: #00ffcc;
      --tech-purple: #7877c6;
    }

    *{ box-sizing:border-box }

    /* Dark mode tech theme */
    @media (prefers-color-scheme: dark) {
      body{
        margin:0; font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color:#e8eaed;
        position: relative;
        overflow-x: hidden;
      }

      body::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
          radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
          radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
          radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
        animation: techPulse 8s ease-in-out infinite;
        z-index: -1;
      }
    }

    /* Light mode tech theme */
    @media (prefers-color-scheme: light) {
      body{
        margin:0; font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
        color:#1e293b;
        position: relative;
        overflow-x: hidden;
      }

      body::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
          radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 80% 20%, rgba(0, 102, 255, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 40% 40%, rgba(0, 255, 204, 0.08) 0%, transparent 50%);
        animation: techPulse 8s ease-in-out infinite;
        z-index: -1;
      }
    }

    @keyframes techPulse {
      0%, 100% { opacity: 0.5; }
      50% { opacity: 0.8; }
    }

    @keyframes techGradient {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    /* Dark mode header */
    @media (prefers-color-scheme: dark) {
      header{
        padding:20px 24px; position:sticky; top:0;
        background:linear-gradient(180deg,rgba(10,10,10,0.9) 60%,rgba(10,10,10,0));
        z-index:2; border-bottom:1px solid rgba(0, 212, 255, 0.2);
        backdrop-filter: blur(10px);
      }

      h1{
        margin:0; font-size:22px;
        background: linear-gradient(45deg, var(--tech-blue), var(--tech-cyan), var(--tech-purple));
        background-size: 400% 400%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: techGradient 3s ease infinite;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
      }

      .muted{ opacity:.7; font-size:13px; color: #a0a0a0; }

      #backBtn{
        display:none; padding:6px 10px; border-radius:10px;
        border:1px solid rgba(0, 212, 255, 0.3);
        background:rgba(0, 0, 0, 0.3);
        color:#e8eaed; cursor:pointer;
        transition: all 0.3s ease;
      }

      #backBtn:hover {
        border-color: var(--tech-blue);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
      }

      #search{
        flex:1 1 260px; min-width:240px; max-width:520px;
        padding:10px 14px; border-radius:12px;
        border:1px solid rgba(0, 212, 255, 0.3);
        background:rgba(0, 0, 0, 0.3);
        color:#e8eaed; outline:none;
        transition: all 0.3s ease;
      }

      #search:focus {
        border-color: var(--tech-blue);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
      }
    }

    /* Light mode header */
    @media (prefers-color-scheme: light) {
      header{
        padding:20px 24px; position:sticky; top:0;
        background:linear-gradient(180deg,rgba(248,250,252,0.9) 60%,rgba(248,250,252,0));
        z-index:2; border-bottom:1px solid rgba(0, 212, 255, 0.3);
        backdrop-filter: blur(10px);
      }

      h1{
        margin:0; font-size:22px;
        background: linear-gradient(45deg, #0066ff, #00d4ff, #00ffcc);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: techGradient 3s ease infinite;
        text-shadow: 0 0 20px rgba(0, 102, 255, 0.3);
      }

      .muted{ opacity:.7; font-size:13px; color: #64748b; }

      #backBtn{
        display:none; padding:6px 10px; border-radius:10px;
        border:1px solid rgba(0, 212, 255, 0.4);
        background:rgba(255, 255, 255, 0.8);
        color:#1e293b; cursor:pointer;
        transition: all 0.3s ease;
      }

      #backBtn:hover {
        border-color: #0066ff;
        box-shadow: 0 0 10px rgba(0, 102, 255, 0.3);
      }

      #search{
        flex:1 1 260px; min-width:240px; max-width:520px;
        padding:10px 14px; border-radius:12px;
        border:1px solid rgba(0, 212, 255, 0.4);
        background:rgba(255, 255, 255, 0.8);
        color:#1e293b; outline:none;
        transition: all 0.3s ease;
      }

      #search:focus {
        border-color: #0066ff;
        box-shadow: 0 0 10px rgba(0, 102, 255, 0.3);
      }
    }

    .row{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; justify-content:center; }

    main{ padding:16px 24px 24px; display:grid; place-items:center; }

    .group-wrap{ width:min(900px,100%); }
    .group-list{ list-style:none; margin:0; padding:0; display:grid; gap:10px; }

    /* Dark mode cards */
    @media (prefers-color-scheme: dark) {
      .group-item{
        display:flex; align-items:center; gap:12px; padding:12px 14px;
        background:rgba(0, 0, 0, 0.3); border:1px solid rgba(0, 212, 255, 0.2); border-radius:14px; cursor:pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
      }
      .group-item:hover{
        transform: translateY(-1px);
        border-color:var(--tech-blue);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
      }

      .card{
        background:rgba(0, 0, 0, 0.3); border:1px solid rgba(0, 212, 255, 0.2); border-radius:var(--card-radius);
        overflow:hidden; box-shadow:var(--shadow);
        transition:all 0.3s ease; cursor:pointer; display:flex; flex-direction:column; max-width:var(--maxW);
        backdrop-filter: blur(10px);
      }
      .card:hover{
        transform:translateY(-2px);
        border-color:var(--tech-blue);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2);
      }
      .thumb-box{
        position:relative; width:100%; aspect-ratio:2/1;
        background:linear-gradient(135deg, #0e121b 0%, #1a1a2e 100%);
        display:grid; place-items:center; overflow:hidden;
        border-bottom: 1px solid rgba(0, 212, 255, 0.1);
      }
      .open{
        font-size:12px; opacity:.7; padding:6px 8px;
        border:1px solid rgba(0, 212, 255, 0.3);
        border-radius:10px;
        background:rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
      }
      .open:hover {
        background:rgba(0, 212, 255, 0.2);
        border-color: var(--tech-blue);
      }
    }

    /* Light mode cards */
    @media (prefers-color-scheme: light) {
      .group-item{
        display:flex; align-items:center; gap:12px; padding:12px 14px;
        background:rgba(255, 255, 255, 0.8); border:1px solid rgba(0, 212, 255, 0.3); border-radius:14px; cursor:pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      }
      .group-item:hover{
        transform: translateY(-1px);
        border-color:#0066ff;
        box-shadow: 0 4px 15px rgba(0, 102, 255, 0.2);
      }

      .card{
        background:rgba(255, 255, 255, 0.8); border:1px solid rgba(0, 212, 255, 0.3); border-radius:var(--card-radius);
        overflow:hidden; box-shadow:0 4px 6px rgba(0, 0, 0, 0.1);
        transition:all 0.3s ease; cursor:pointer; display:flex; flex-direction:column; max-width:var(--maxW);
        backdrop-filter: blur(10px);
      }
      .card:hover{
        transform:translateY(-2px);
        border-color:#0066ff;
        box-shadow: 0 8px 25px rgba(0, 102, 255, 0.2);
      }
      .thumb-box{
        position:relative; width:100%; aspect-ratio:2/1;
        background:linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        display:grid; place-items:center; overflow:hidden;
        border-bottom: 1px solid rgba(0, 212, 255, 0.2);
      }
      .open{
        font-size:12px; opacity:.7; padding:6px 8px;
        border:1px solid rgba(0, 212, 255, 0.4);
        border-radius:10px;
        background:rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
      }
      .open:hover {
        background:rgba(0, 212, 255, 0.2);
        border-color: #0066ff;
      }
    }

    .gname{ font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; width:100%; }
    .grid{
      width:min(1200px,100%);
      display:grid;
      grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
      gap:var(--gap);
      align-items:start;
      justify-items:stretch;
      margin: 0 auto;
      padding: 0 20px;
    }
    .thumb{ max-width:100%; max-height:100%; object-fit:contain; display:block; }
    .meta{ padding:12px 14px; display:flex; justify-content:space-between; align-items:center; gap:8px; }
    .title{ font-weight:600; font-size:14px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .empty{ opacity:.6; padding:40px 0; text-align:center; }
    .crumb{ font-size:13px; opacity:.8; }

    .overlay{ position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; place-items:center; padding:20px; z-index:10; }
    .overlay.show{ display:grid; }

    /* Dark mode viewer */
    @media (prefers-color-scheme: dark) {
      .viewer{
        inline-size:min(92vw,var(--maxW));
        block-size:min(82vh,var(--maxH));
        background:#0e121b; border:1px solid rgba(0, 212, 255, 0.3); border-radius:18px; overflow:hidden; position:relative; box-shadow:0 12px 36px rgba(0,0,0,.35);
        display:grid;
      }
      .chip{ background:rgba(0,0,0,.45); border:1px solid rgba(0, 212, 255, 0.3); color:#e8eaed; padding:6px 10px; border-radius:12px; font-size:12px; max-width:60%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .btn{ margin-left:auto; background:rgba(0, 0, 0, 0.3); color:#e8eaed; border:1px solid rgba(0, 212, 255, 0.3); border-radius:10px; padding:6px 10px; cursor:pointer; transition: all 0.3s ease; }
      .btn:hover { border-color: var(--tech-blue); box-shadow: 0 0 10px rgba(0, 212, 255, 0.3); }
      .mv-box{ width:100%; aspect-ratio:1036/518; background:#0b0d12; border:1px solid rgba(0, 212, 255, 0.2); border-radius:12px; overflow:hidden; }
      .mv-box model-viewer{ width:100%; height:100%; background:#0b0d12; }
      .res-cell{ position:relative; width:100%; aspect-ratio:2/1; background:#0e121b; border:1px solid rgba(0, 212, 255, 0.2); border-radius:12px; overflow:hidden; display:grid; place-items:center; }
      .res-empty{ position:absolute; inset:0; display:grid; place-items:center; opacity:.55; font-size:12px; color:#9aa0a6; }
      .download-icon{ background:rgba(0, 0, 0, 0.6); border:1px solid rgba(0, 212, 255, 0.3); color:#e8eaed; box-shadow:0 4px 12px rgba(0,0,0,0.3); }
      .download-icon:hover{ background:rgba(0, 212, 255, 0.2); border-color:var(--tech-blue); box-shadow:0 0 20px rgba(0, 212, 255, 0.4); transform:scale(1.05); }
    }

    /* Light mode viewer */
    @media (prefers-color-scheme: light) {
      .viewer{
        inline-size:min(92vw,var(--maxW));
        block-size:min(82vh,var(--maxH));
        background:#f8fafc; border:1px solid rgba(0, 212, 255, 0.4); border-radius:18px; overflow:hidden; position:relative; box-shadow:0 12px 36px rgba(0,0,0,.15);
        display:grid;
      }
      .chip{ background:rgba(255,255,255,0.8); border:1px solid rgba(0, 212, 255, 0.4); color:#1e293b; padding:6px 10px; border-radius:12px; font-size:12px; max-width:60%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .btn{ margin-left:auto; background:rgba(255, 255, 255, 0.8); color:#1e293b; border:1px solid rgba(0, 212, 255, 0.4); border-radius:10px; padding:6px 10px; cursor:pointer; transition: all 0.3s ease; }
      .btn:hover { border-color: #0066ff; box-shadow: 0 0 10px rgba(0, 102, 255, 0.3); }
      .mv-box{ width:100%; aspect-ratio:1036/518; background:#f8fafc; border:1px solid rgba(0, 212, 255, 0.3); border-radius:12px; overflow:hidden; }
      .mv-box model-viewer{ width:100%; height:100%; background:#f8fafc; }
      .res-cell{ position:relative; width:100%; aspect-ratio:2/1; background:#f8fafc; border:1px solid rgba(0, 212, 255, 0.3); border-radius:12px; overflow:hidden; display:grid; place-items:center; }
      .res-empty{ position:absolute; inset:0; display:grid; place-items:center; opacity:.55; font-size:12px; color:#64748b; }
      .download-icon{ background:rgba(255, 255, 255, 0.9); border:1px solid rgba(0, 212, 255, 0.4); color:#1e293b; box-shadow:0 4px 12px rgba(0,0,0,0.15); }
      .download-icon:hover{ background:rgba(0, 212, 255, 0.2); border-color:#0066ff; box-shadow:0 0 20px rgba(0, 102, 255, 0.4); transform:scale(1.05); }
    }

    .viewer-header{ position:absolute; top:8px; left:8px; right:8px; display:flex; gap:8px; align-items:center; z-index:2; }
    .viewer-body{ height:100%; display:grid; grid-template-rows:auto auto; gap:12px; padding:36px 8px 8px 8px; overflow:auto; }
    .res-grid{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .res-img{ max-width:100%; max-height:100%; object-fit:contain; display:block; cursor:zoom-in; }
    .res-cell:has(.res-img){ cursor:zoom-in; transition:border-color .2s ease, transform .2s ease; }
    .res-cell:has(.res-img):hover{ border-color:var(--tech-blue); transform:translateY(-1px); }
    .image-overlay{ position:fixed; inset:0; z-index:30; display:none; place-items:center; background:rgba(0,0,0,.9); padding:24px; }
    .image-overlay.show{ display:grid; }
    .image-stage{ position:relative; width:min(96vw,1600px); height:min(92vh,1100px); display:grid; place-items:center; }
    .image-stage img{ max-width:100%; max-height:100%; object-fit:contain; border-radius:10px; box-shadow:0 18px 60px rgba(0,0,0,.5); }
    .image-close,.image-nav{ position:absolute; z-index:2; border:1px solid rgba(0,212,255,.45); background:rgba(5,10,18,.8); color:#fff; cursor:pointer; display:grid; place-items:center; }
    .image-close{ top:8px; right:8px; width:44px; height:44px; border-radius:50%; font-size:26px; }
    .image-nav{ top:50%; translate:0 -50%; width:48px; height:64px; border-radius:12px; font-size:28px; }
    .image-prev{ left:8px; }
    .image-next{ right:8px; }
    .image-nav:disabled{ opacity:.25; cursor:default; }
    .image-caption{ position:absolute; left:50%; bottom:8px; translate:-50% 0; padding:7px 12px; border-radius:10px; background:rgba(5,10,18,.78); color:#fff; font-size:13px; white-space:nowrap; }
    .download-icon{ position:absolute; bottom:16px; right:16px; width:44px; height:44px; border-radius:50%; display:grid; place-items:center; font-size:20px; cursor:pointer; z-index:3; transition:all 0.3s ease; }

    /* Pagination controls */
    .pager {
      grid-column: 1 / -1;
      justify-content: center;
      align-items: center;
      display: flex;
      gap: 16px;
      margin-top: 8px;
      font-size: 13px;
      text-align: center;
    }

    /* Dark mode pagination */
    @media (prefers-color-scheme: dark) {
      .pager {
        color: #ccc;
      }
      .pager button {
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        background: rgba(0, 0, 0, 0.3);
        color: #e8eaed;
        cursor: pointer;
        transition: all 0.3s ease;
      }
      .pager button:hover:not(:disabled) {
        border-color: var(--tech-blue);
        box-shadow: 0 0 8px rgba(0, 212, 255, 0.2);
      }
      .pager button:disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }
    }

    /* Light mode pagination */
    @media (prefers-color-scheme: light) {
      .pager {
        color: #64748b;
      }
      .pager button {
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid rgba(0, 212, 255, 0.4);
        background: rgba(255, 255, 255, 0.8);
        color: #1e293b;
        cursor: pointer;
        transition: all 0.3s ease;
      }
      .pager button:hover:not(:disabled) {
        border-color: #0066ff;
        box-shadow: 0 0 8px rgba(0, 102, 255, 0.2);
      }
      .pager button:disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }
    }

    /* Intro card styles */
    @media (prefers-color-scheme: dark) {
      .intro-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 102, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        backdrop-filter: blur(10px);
      }
      .intro-title {
        background: linear-gradient(45deg, var(--tech-blue), var(--tech-cyan), var(--tech-purple));
        background-size: 400% 400%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: techGradient 3s ease infinite;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
      }
      .intro-description {
        color: #e0e0e0;
      }
    }

    @media (prefers-color-scheme: light) {
      .intro-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(0, 102, 255, 0.05) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }
      .intro-title {
        background: linear-gradient(45deg, #0066ff, #00d4ff, #00ffcc);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: techGradient 3s ease infinite;
        text-shadow: 0 0 15px rgba(0, 102, 255, 0.2);
      }
      .intro-description {
        color: #334155;
      }
    }

    footer{
      opacity:.55;
      font-size:12px;
      padding:12px 24px 24px;
      text-align:center;
      display:flex;
      justify-content:center;
      align-items:center;
      width:100%;
    }
  </style>
</head>
<body>
  <header>
    <div class="row">
      <button id="backBtn">← Back</button>
      <h1 id="pageTitle">Depth Anything 3 Gallery</h1>
      <span id="crumb" class="crumb"></span>
      <input id="search" placeholder="Search…" />
    </div>
    <div class="muted" id="hint" style="text-align: center;">Level 1 shows groups only; click a group to browse scenes and previews.</div>
  </header>

  <main>
    <!-- Tech intro card -->
    <div class="intro-card" style="margin-bottom: 30px; padding: 25px; border-radius: 15px; text-align: center; max-width: 800px;">
      <h2 class="intro-title" style="margin: 0 0 15px 0; font-size: 1.8em; font-weight: 700;">
        🎯 Depth Anything 3 Gallery
      </h2>
      <p class="intro-description" style="margin: 0; font-size: 1.1em; line-height: 1.6;">
        Explore 3D reconstructions and depth visualizations from Depth Anything 3.
        Browse through groups of scenes, preview 3D models, and examine depth maps interactively.
      </p>
    </div>

    <div id="level1" class="group-wrap" aria-live="polite">
      <ul id="groupList" class="group-list"></ul>
      <div id="groupEmpty" class="empty" style="display:none;">No available groups</div>
    </div>

    <div id="level2" style="display:none; width:100%;" aria-live="polite">
      <div id="topPager" class="pager" style="margin-bottom: 16px;"></div>
      <div id="grid" class="grid"></div>
      <div id="sceneEmpty" class="empty" style="display:none;">No available scenes in this group</div>
    </div>
  </main>

  <div id="overlay" class="overlay" role="dialog" aria-modal="true" aria-label="3D Preview">
    <div class="viewer" id="viewer">
      <div class="viewer-header">
        <div id="viewerTitle" class="chip">Loading…</div>
        <button id="toggleView" class="btn" title="Toggle between 3D-only and resource view">Resource View</button>
        <button id="closeBtn" class="btn">Close</button>
      </div>
      <div id="downloadBtn" class="download-icon" title="Download GLB model">⬇</div>
      <div class="viewer-body">
        <div class="mv-box"><model-viewer id="mv"
          src=""
          ar
          camera-controls
          auto-rotate
          interaction-prompt="auto"
          shadow-intensity="0.7"
          exposure="1.0"
          alt="GLB Preview"></model-viewer></div>
        <div class="res-grid" id="resGrid" hidden></div>
      </div>
    </div>
  </div>

  <div id="imageOverlay" class="image-overlay" role="dialog" aria-modal="true" aria-label="Expanded depth image">
    <div class="image-stage">
      <button id="imageClose" class="image-close" aria-label="Close expanded image">×</button>
      <button id="imagePrev" class="image-nav image-prev" aria-label="Previous image">‹</button>
      <img id="expandedImage" src="" alt="" />
      <button id="imageNext" class="image-nav image-next" aria-label="Next image">›</button>
      <div id="imageCaption" class="image-caption"></div>
    </div>
  </div>

  <footer>Depth Anything 3 Gallery. Copyright 2025 Depth Anything 3 authors.</footer>

<script>
const level1=document.getElementById('level1'),level2=document.getElementById('level2'),pageTitle=document.getElementById('pageTitle'),crumb=document.getElementById('crumb'),backBtn=document.getElementById('backBtn'),hint=document.getElementById('hint'),searchInput=document.getElementById('search'),groupList=document.getElementById('groupList'),groupEmpty=document.getElementById('groupEmpty'),topPager=document.getElementById('topPager'),grid=document.getElementById('grid'),sceneEmpty=document.getElementById('sceneEmpty'),overlay=document.getElementById('overlay'),viewer=document.getElementById('viewer'),mv=document.getElementById('mv'),viewerTitle=document.getElementById('viewerTitle'),downloadBtn=document.getElementById('downloadBtn'),toggleViewBtn=document.getElementById('toggleView'),closeBtn=document.getElementById('closeBtn'),resGrid=document.getElementById('resGrid'),imageOverlay=document.getElementById('imageOverlay'),expandedImage=document.getElementById('expandedImage'),imageCaption=document.getElementById('imageCaption'),imageClose=document.getElementById('imageClose'),imagePrev=document.getElementById('imagePrev'),imageNext=document.getElementById('imageNext');
let GROUPS=[],SCENES=[],currentGroup=null,currentScene=null,currentPage=1,currentScenePage=1,expandedImages=[],expandedIndex=0;

const qs=()=>new URLSearchParams(location.search);
async function loadGroups(){const r=await fetch('/manifest.json',{cache:'no-store'});if(!r.ok)throw new Error(r.status+' '+r.statusText);const j=await r.json();GROUPS=j.groups||[];renderGroups(GROUPS);}
async function loadScenes(g){const r=await fetch('/manifest/'+encodeURIComponent(g)+'.json',{cache:'no-store'});if(!r.ok)throw new Error(r.status+' '+r.statusText);const j=await r.json();SCENES=j.items||[];const p=parseInt(qs().get('page'))||1;renderScenes(SCENES,p);}
function renderGroups(list){groupList.innerHTML='';const q=searchInput.value.trim().toLowerCase();const f=list.filter(g=>(g.title||g.id||'').toLowerCase().includes(q));if(!f.length){groupEmpty.style.display='';return;}groupEmpty.style.display='none';for(const g of f){const li=document.createElement('li');li.className='group-item';li.title=g.title||g.id;li.onclick=()=>enterLevel2(g.id,{push:true});const name=document.createElement('div');name.className='gname';name.textContent=g.title||g.id;li.appendChild(name);groupList.appendChild(li);}}
function renderScenes(list,page=1){topPager.innerHTML='';grid.innerHTML='';const q=searchInput.value.trim().toLowerCase();const f=list.filter(x=>(x.title||'').toLowerCase().includes(q)||(x.id||'').toLowerCase().includes(q));if(!f.length){sceneEmpty.style.display='';topPager.style.display='none';return;}sceneEmpty.style.display='none';topPager.style.display='flex';const perPage=16;const total=f.length;const totalPages=Math.max(1,Math.ceil(total/perPage));currentScenePage=page;const u=new URL(location.href);u.searchParams.set('page',page);history.replaceState(null,'',u);const subset=f.slice((page-1)*perPage,page*perPage);for(const i of subset){const c=document.createElement('div');c.className='card';c.title=i.title;const b=document.createElement('div');b.className='thumb-box';const img=document.createElement('img');img.className='thumb';img.loading='lazy';img.alt=i.title;img.src=i.thumbnail;b.appendChild(img);const m=document.createElement('div');m.className='meta';const t=document.createElement('div');t.className='title';t.textContent=i.title;const o=document.createElement('div');o.className='open';o.textContent='Preview';m.appendChild(t);m.appendChild(o);c.appendChild(b);c.appendChild(m);c.onclick=()=>openViewer(i,{push:true});grid.appendChild(c);}function buildPager(){const pg=document.createElement('div');pg.className='pager';const prev=document.createElement('button');prev.textContent='← Prev';prev.disabled=page<=1;prev.onclick=()=>renderScenes(list,page-1);const info=document.createElement('span');info.textContent=`${page} / ${totalPages}`;const next=document.createElement('button');next.textContent='Next →';next.disabled=page>=totalPages;next.onclick=()=>renderScenes(list,page+1);pg.appendChild(prev);pg.appendChild(info);pg.appendChild(next);return pg;}topPager.innerHTML='';topPager.appendChild(buildPager());grid.appendChild(buildPager());}
function enterLevel1({push=false}={}){currentGroup=null;pageTitle.textContent='Depth Anything 3 Gallery';crumb.textContent='';backBtn.style.display='none';hint.style.display='';level1.style.display='';level2.style.display='none';overlay.classList.remove('show');mv.src='';const u=new URL(location.href);u.searchParams.delete('group');u.searchParams.delete('id');u.searchParams.delete('page');push?history.pushState(null,'',u):history.replaceState(null,'',u);searchInput.value='';loadGroups().catch(e=>{groupList.innerHTML='';groupEmpty.style.display='';groupEmpty.textContent='Failed to load groups: '+e;});}
async function enterLevel2(g,{push=false}={}){currentGroup=g;pageTitle.textContent=g;crumb.textContent='(group)';backBtn.style.display='';hint.style.display='none';level1.style.display='none';level2.style.display='';overlay.classList.remove('show');mv.src='';const u=new URL(location.href);u.searchParams.set('group',g);u.searchParams.delete('id');push?history.pushState(null,'',u):history.replaceState(null,'',u);searchInput.value='';try{await loadScenes(g);const id=qs().get('id');if(id){const hit=SCENES.find(x=>x.id===id);if(hit)openViewer(hit,{push:false});}}catch(e){grid.innerHTML='';sceneEmpty.style.display='';sceneEmpty.textContent='Failed to load scenes: '+e;}}
function buildResGrid(i,page=1){
  resGrid.innerHTML='';
  const imgs=i.depth_images||[];
  const perPage=4;
  const total=imgs.length;
  const totalPages=Math.max(1, Math.ceil(total/perPage));
  currentPage=page;

  const subset=imgs.slice((page-1)*perPage,(page-1)*perPage+perPage);
  for(let k=0;k<4;k++){
    const cell=document.createElement('div');
    cell.className='res-cell';
    if(subset[k]){
      const im=document.createElement('img');
      im.className='res-img';
      im.src=subset[k];
      im.alt=(i.title||'scene')+' depth '+(k+1+(page-1)*perPage);
      im.loading='lazy';
      im.onclick=()=>openExpandedImage(i,k+(page-1)*perPage);
      cell.appendChild(im);
    } else {
      const ph=document.createElement('div');
      ph.className='res-empty';
      ph.textContent='N/A';
      cell.appendChild(ph);
    }
    resGrid.appendChild(cell);
  }

  // pagination bar (always rebuilt)
  const pager=document.createElement('div');
  pager.className='pager';

  const prev=document.createElement('button');
  prev.textContent='← Prev';
  prev.disabled=page<=1;
  prev.onclick=()=>buildResGrid(i,page-1);

  const info=document.createElement('span');
  info.textContent=`${page} / ${totalPages}`;

  const next=document.createElement('button');
  next.textContent='Next →';
  next.disabled=page>=totalPages;
  next.onclick=()=>buildResGrid(i,page+1);

  pager.appendChild(prev);
  pager.appendChild(info);
  pager.appendChild(next);
  resGrid.appendChild(pager);
}
function openExpandedImage(i,index){expandedImages=i.depth_images||[];expandedIndex=index;renderExpandedImage();imageOverlay.classList.add('show');imageClose.focus();}
function renderExpandedImage(){const src=expandedImages[expandedIndex];if(!src)return;expandedImage.src=src;expandedImage.alt=(currentScene?.title||'scene')+' depth '+(expandedIndex+1);imageCaption.textContent=`${currentScene?.title||'Scene'} · ${expandedIndex+1} / ${expandedImages.length}`;imagePrev.disabled=expandedIndex<=0;imageNext.disabled=expandedIndex>=expandedImages.length-1;}
function stepExpandedImage(delta){const next=expandedIndex+delta;if(next<0||next>=expandedImages.length)return;expandedIndex=next;renderExpandedImage();}
function closeExpandedImage(){imageOverlay.classList.remove('show');expandedImage.src='';}
function openViewer(i,{push=false}={}){currentScene=i;viewerTitle.textContent=i.title;mv.src=i.model;overlay.classList.add('show');resGrid.hidden=true;toggleViewBtn.textContent='Resource View';viewer.style.blockSize='min(82vh,var(--maxH))';buildResGrid(i,1);downloadBtn.onclick=()=>{const a=document.createElement('a');a.href=i.model;a.download=i.title+'.glb';a.click();};if(push){const u=new URL(location.href);if(!u.searchParams.get('group'))u.searchParams.set('group',currentGroup||'');u.searchParams.set('id',i.id);history.pushState(null,'',u);}}
function toggleView(){const hidden=!resGrid.hidden;resGrid.hidden=hidden;toggleViewBtn.textContent=hidden?'Resource View':'3D Only';viewer.style.blockSize=hidden?'min(82vh,var(--maxH))':'min(92vh,900px)';}
function closeViewer(){const hasId=!!qs().get('id');if(hasId&&history.length>1){history.back();return;}const u=new URL(location.href);u.searchParams.delete('id');history.replaceState(null,'',u);overlay.classList.remove('show');mv.src='';}
overlay.onclick=e=>{if(e.target===overlay)closeViewer();};closeBtn.onclick=closeViewer;toggleViewBtn.onclick=toggleView;backBtn.onclick=()=>history.back();
imageOverlay.onclick=e=>{if(e.target===imageOverlay)closeExpandedImage();};imageClose.onclick=closeExpandedImage;imagePrev.onclick=()=>stepExpandedImage(-1);imageNext.onclick=()=>stepExpandedImage(1);
document.addEventListener('keydown',e=>{if(!imageOverlay.classList.contains('show'))return;if(e.key==='Escape')closeExpandedImage();else if(e.key==='ArrowLeft')stepExpandedImage(-1);else if(e.key==='ArrowRight')stepExpandedImage(1);});
searchInput.oninput=()=>{!qs().get('group')?renderGroups(GROUPS):renderScenes(SCENES,1);};
window.onpopstate=()=>routeFromURL();
async function routeFromURL(){if(location.pathname!="/")history.replaceState(null,'','/'+location.search);const g=qs().get('group');const id=qs().get('id');if(!g){enterLevel1({push:false});return;}await enterLevel2(g,{push:false});if(id){const hit=SCENES.find(x=>x.id===id);if(hit)openViewer(hit,{push:false});else{overlay.classList.remove('show');mv.src='';}}else{overlay.classList.remove('show');mv.src='';}}
routeFromURL();
</script>
</body>
</html>
"""

# ------------------------------ Utilities ------------------------------ #

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _url_join(*parts: str) -> str:
    norm = posixpath.join(*[p.replace("\\", "/") for p in parts])
    segs = [s for s in norm.split("/") if s not in ("", ".")]
    return "/".join(quote(s) for s in segs)


def _is_plain_name(name: str) -> bool:
    return all(c not in name for c in ("/", "\\")) and name not in (".", "..")


def build_group_list(root_dir: str) -> dict:
    groups = []
    try:
        for gname in sorted(os.listdir(root_dir)):
            gpath = os.path.join(root_dir, gname)
            if not os.path.isdir(gpath):
                continue
            has_scene = False
            try:
                for sname in os.listdir(gpath):
                    spath = os.path.join(gpath, sname)
                    if not os.path.isdir(spath):
                        continue
                    if os.path.exists(os.path.join(spath, "scene.glb")) and os.path.exists(
                        os.path.join(spath, "scene.jpg")
                    ):
                        has_scene = True
                        break
            except Exception:
                pass
            if has_scene:
                groups.append({"id": gname, "title": gname})
    except Exception as e:
        print(f"[warn] build_group_list failed: {e}", file=sys.stderr)
    return {"groups": groups}


def build_group_manifest(root_dir: str, group: str) -> dict:
    items = []
    gpath = os.path.join(root_dir, group)
    try:
        if not os.path.isdir(gpath):
            return {"group": group, "items": []}
        for sname in sorted(os.listdir(gpath)):
            spath = os.path.join(gpath, sname)
            if not os.path.isdir(spath):
                continue
            glb_fs = os.path.join(spath, "scene.glb")
            jpg_fs = os.path.join(spath, "scene.jpg")
            if not (os.path.exists(glb_fs) and os.path.exists(jpg_fs)):
                continue
            depth_images = []
            dpath = os.path.join(spath, "depth_vis")
            if os.path.isdir(dpath):
                files = [
                    f for f in os.listdir(dpath) if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                ]
                for fn in sorted(files):
                    depth_images.append("/" + _url_join(group, sname, "depth_vis", fn))
            items.append(
                {
                    "id": sname,
                    "title": sname,
                    "model": "/" + _url_join(group, sname, "scene.glb"),
                    "thumbnail": "/" + _url_join(group, sname, "scene.jpg"),
                    "depth_images": depth_images,
                }
            )
    except Exception as e:
        print(f"[warn] build_group_manifest failed for {group}: {e}", file=sys.stderr)
    return {"group": group, "items": items}


class GalleryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html") or self.path.startswith("/?"):
            content = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        if self.path == "/manifest.json":
            data = json.dumps(
                build_group_list(self.directory), ensure_ascii=False, indent=2
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/manifest/") and self.path.endswith(".json"):
            group_enc = self.path[len("/manifest/") : -len(".json")]
            try:
                group = unquote(group_enc)
            except Exception:
                group = group_enc
            if not _is_plain_name(group):
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid group name")
                return
            data = json.dumps(
                build_group_manifest(self.directory, group), ensure_ascii=False, indent=2
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        return super().do_GET()

    def list_directory(self, path):
        self.send_error(HTTPStatus.NOT_FOUND, "Directory listing disabled")
        return None


def gallery():
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 Gallery Server (two-level, with pagination)"
    )
    parser.add_argument(
        "-d", "--dir", required=True, help="Gallery root directory (two-level: group/scene)"
    )
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port (default 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default 127.0.0.1)")
    parser.add_argument("--open", action="store_true", help="Open browser after launch")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.dir)
    if not os.path.isdir(root_dir):
        print(f"[error] Directory not found: {root_dir}", file=sys.stderr)
        sys.exit(1)

    Handler = partial(GalleryHandler, directory=root_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    addr = f"http://{args.host}:{args.port}/"
    print(f"[info] Serving gallery from: {root_dir}")
    print(f"[info] Open: {addr}")

    if args.open:
        try:
            import webbrowser

            webbrowser.open(addr)
        except Exception as e:
            print(f"[warn] Failed to open browser: {e}", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] Shutting down...")
    finally:
        server.server_close()


def main():
    """Main entry point for gallery server."""
    mimetypes.add_type("model/gltf-binary", ".glb")
    gallery()


if __name__ == "__main__":
    main()
