/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

import { Link } from 'react-router-dom';
import { Search } from 'lucide-react';

export default function Header() {
  return (
    <header className="bg-white shadow">
      <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
        <Link to="/" className="text-2xl font-bold text-blue-600">
          🎬 Searvids
        </Link>
        <nav className="flex gap-6">
          <Link to="/" className="text-gray-700 hover:text-blue-600">
            Home
          </Link>
          <Link to="/search" className="text-gray-700 hover:text-blue-600 flex items-center gap-2">
            <Search size={20} />
            Search
          </Link>
        </nav>
      </div>
    </header>
  );
}