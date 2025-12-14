/*
 * Copyright (c) 2025 Hyeonjun Park (phjun7150@gmail.com)
 * All rights reserved.
 */

import { Link } from 'react-router-dom';
import { Search, Video } from 'lucide-react';

export default function Header() {
  return (
    <header className="bg-slate-800 border-b border-slate-700 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 text-xl font-bold text-blue-400 hover:text-blue-300 transition-colors">
          <Video className="w-6 h-6" />
          <span>Searvids</span>
        </Link>
        <nav className="flex gap-6">
          <Link to="/" className="text-slate-300 hover:text-white font-medium transition-colors">
            Home
          </Link>
          <Link to="/search" className="text-slate-300 hover:text-white font-medium flex items-center gap-2 transition-colors">
            <Search size={18} />
            Search
          </Link>
        </nav>
      </div>
    </header>
  );
}