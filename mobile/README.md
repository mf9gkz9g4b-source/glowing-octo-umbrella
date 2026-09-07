# Mobile app folder (Capacitor) - setup instructions

This folder holds the Capacitor scaffold and helper scripts to create a mobile app from the web UI in ../static.

What this does
- Keeps a separate mobile workspace so you can build native mobile apps without mixing root-level files.
- The web assets are copied from the main `static/` folder into `mobile/www` during setup.

Quick start (macOS / Linux)
1. From repo root, run the helper script (it will copy web assets and install npm deps):

   cd mobile
   ./setup.sh

2. Add Android or iOS platform (first time only):

   npx cap add android
   npx cap add ios   # macOS only

3. Copy web assets into native projects and open IDE:

   npx cap copy
   npx cap open android   # opens Android Studio
   npx cap open ios       # opens Xcode (macOS only)

Notes
- The script will not commit or push changes. It configures the mobile folder on your machine.
- You must have Node.js, npm, Java JDK, and Android Studio for native builds.
- Do NOT copy any secrets into mobile/www. The app should call your server (hosted separately) which holds private API keys.
