/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

#include <crow/app.h>
#include "server_api.h"
#include "hnsw_index.h"
#include "utils.h"
#include <iostream>

int main(int argc, char** argv) {
    // 1. Initialize HNSW index
    hnsw_index::create(512, "cosine");

    // 2. Create Crow app
    crow::SimpleApp app;

    // 3. Register API routes
    server_api::setup_routes(app);

    // 4. Server start log
    utils::Logger::instance().info("🚀 SearvidsServer started on http://localhost:8080");

    // 5. Run server (port 8080, multithreaded)
    app.port(8080).multithreaded().run();

    return 0;
}