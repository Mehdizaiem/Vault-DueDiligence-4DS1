"use client";

import { useEffect, useRef } from 'react';
import Link from 'next/link';
import * as THREE from 'three';
import { 
  FileText, 
  HelpCircle, 
  AlertTriangle,
  TrendingUp,
  BarChart4,
  Newspaper,
  ShieldCheck,
  ArrowRight,
  Zap,
  Lock,
  TrendingDown,
  Crown,
  Sparkles,
  Star,
  ChevronRight
} from "lucide-react";

const features = [
  {
    label: 'Document Analysis',
    icon: FileText,
    color: "text-violet-500",
    bgColor: "bg-violet-50",
    description: "Analyze fund documentation automatically",
    href: "/dashboard/documents"
  },
  {
    label: 'Q&A System',
    icon: HelpCircle,
    color: "text-pink-500",
    bgColor: "bg-pink-50",
    description: "Get instant answers to due diligence queries",
    href: "/dashboard/qa"
  },
  {
    label: 'Risk Analysis',
    icon: AlertTriangle,
    color: "text-orange-500",
    bgColor: "bg-orange-50",
    description: "Monitor real-time risk metrics and alerts",
    href: "/dashboard/risk"
  },
  {
    label: 'Crypto Forecast',
    icon: TrendingUp,
    color: "text-green-500",
    bgColor: "bg-green-50",
    description: "Price predictions and market analysis",
    href: "/dashboard/forecast"
  },
  {
    label: 'Analytics',
    icon: BarChart4,
    color: "text-indigo-500",
    bgColor: "bg-indigo-50",
    description: "Crypto portfolio insights",
    href: "/dashboard/analytics"
  },
  {
    label: 'News',
    icon: Newspaper,
    color: "text-yellow-500",
    bgColor: "bg-yellow-50",
    description: "Latest market events and updates",
    href: "/dashboard/news"
  }
];

// Added the Report Generator separately so we can handle the layout differently
const reportGeneratorFeature = {
  label: 'Report Generator',
  icon: FileText,
  color: "text-amber-700", // Tailwind's brown-like color
  bgColor: "bg-amber-100", // Tailwind's brown-like color
  description: "Generate compliance reports",
  href: "/dashboard/reports"
};

// Market pulse data - more valuable than generic stats
const marketPulse = [
  {
    label: 'Bitcoin',
    value: '$100,245',
    change: '+2.4%',
    trend: 'up',
    icon: TrendingUp
  },
  {
    label: 'Market Sentiment',
    value: 'Moderate',
    change: '+0.5%',
    icon: Zap
  },
  {
    label: 'Risk Rating',
    value: 'Moderate',
    icon: ShieldCheck
  },
  {
    label: 'Ethereum',
    value: '$3,285',
    change: '-1.2%',
    trend: 'down',
    icon: TrendingDown
  }
];

export default function DashboardPage() {
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    
    let scene: THREE.Scene;
    let camera: THREE.PerspectiveCamera;
    let renderer: THREE.WebGLRenderer;
    const blocks: {
      mesh?: THREE.Mesh;
      initialPosition: THREE.Vector3;
      speed: number;
      angle: number;
      radius?: number;
    }[] = [];
    let animationFrameId: number;
    
    // Initialize Three.js
    const initThree = () => {
      // Scene setup
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
      
      renderer = new THREE.WebGLRenderer({ 
        antialias: true, 
        alpha: true,
        canvas: document.createElement('canvas')
      });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setClearColor(0xffffff, 0); // Transparent background
      renderer.setPixelRatio(window.devicePixelRatio);
      
      // Add renderer to the DOM
      if (canvasRef.current) {
        canvasRef.current.innerHTML = '';
        canvasRef.current.appendChild(renderer.domElement);
      }
      
      // Add subtle lighting
      const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
      scene.add(ambientLight);
      
      const directionalLight = new THREE.DirectionalLight(0x4c6bff, 0.8);
      directionalLight.position.set(0, 1, 1);
      scene.add(directionalLight);
      
      const pointLight = new THREE.PointLight(0x4c6bff, 0.8, 100);
      pointLight.position.set(10, 10, 10);
      scene.add(pointLight);
      
      // Create blockchain visualization
      createBlockchain();
      
      // Position camera
      camera.position.z = 20;
      
      // Set up event listeners
      window.addEventListener('resize', onWindowResize);
      
      // Start animation loop
      animate();
    };
    
    // Create blockchain blocks
    const createBlockchain = () => {
      const blockGeometry = new THREE.BoxGeometry(2, 1, 0.5);
      const blockMaterial = new THREE.MeshPhongMaterial({
        color: 0x4c6bff,
        specular: 0x4c6bff,
        shininess: 30,
        transparent: true,
        opacity: 0.5,
      });
      
      const totalBlocks = 12;
      for (let i = 0; i < totalBlocks; i++) {
        const block = new THREE.Mesh(blockGeometry, blockMaterial.clone());
        
        // Position in a simple circle pattern
        const angle = i * (Math.PI * 2 / totalBlocks);
        const radius = 15;
        
        block.position.x = Math.cos(angle) * radius;
        block.position.y = Math.sin(angle) * radius;
        block.position.z = -5 + Math.random() * 5;
        
        // Random rotation
        block.rotation.x = Math.random() * 0.3 - 0.15;
        block.rotation.y = Math.random() * 0.3 - 0.15;
        block.rotation.z = Math.random() * 0.3 - 0.15;
        
        // Store block data for animation
        blocks.push({
          mesh: block,
          initialPosition: block.position.clone(),
          speed: 0.005 + Math.random() * 0.003,
          angle: angle,
          radius: radius
        });
        
        scene.add(block);
      }
    };
    
    // Update blockchain position and rotation
    const updateBlockchain = (time: number) => {
      // Update blocks
      blocks.forEach((block, index) => {
        if (block.mesh && block.radius !== undefined) {
          // Only animate if it's a regular block with a mesh
          const offset = index * 0.5;
          
          // Update block position in circular pattern
          block.angle += block.speed;
          const newRadius = block.radius + Math.sin(time * 0.0005 + offset) * 0.5;
          
          block.mesh.position.x = Math.cos(block.angle) * newRadius;
          block.mesh.position.y = Math.sin(block.angle) * newRadius;
          
          // Subtle rotation
          block.mesh.rotation.x += 0.003;
          block.mesh.rotation.y += 0.002;
        }
      });
    };
    
    // Animation loop
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      
      // Rotate entire scene slowly for subtle effect
      if (scene) {
        scene.rotation.y += 0.0005;
        scene.rotation.x = Math.sin(Date.now() * 0.0001) * 0.1;
        
        // Update blockchain
        updateBlockchain(Date.now());
        
        // Render scene
        renderer.render(scene, camera);
      }
    };
    
    // Window resize handler
    const onWindowResize = () => {
      if (camera && renderer) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }
    };
    
    // Initialize everything
    initThree();
    
    // Cleanup on unmount
    return () => {
      window.removeEventListener('resize', onWindowResize);
      cancelAnimationFrame(animationFrameId);
      
      // Dispose of resources
      if (renderer) {
        renderer.dispose();
      }
      
      if (scene) {
        scene.clear();
      }
    };
  }, []);

  return (
    <div className="relative min-h-screen flex flex-col overflow-hidden">
      {/* 3D Background Canvas */}
      <div 
        ref={canvasRef} 
        className="fixed inset-0 pointer-events-none z-0 opacity-60"
      />
      
      {/* Content */}
      <div className="relative z-10 flex-1 flex flex-col">
        <div className="max-w-7xl mx-auto w-full px-8 py-6">
        {/* Welcome Banner */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-full p-6 rounded-2xl bg-gradient-to-br from-[#1a1a3a] to-[#0f0f2a] text-white shadow-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex items-center">
                  <h1 className="text-3xl font-bold text-white"
                    style={{
                      textShadow: '0 0 10px rgba(76, 107, 255, 0.8), 0 0 20px rgba(76, 107, 255, 0.4)'
                    }}
                  >
                    VAULT
                  </h1>
                </div>
                <div>
                  <p className="text-lg text-gray-200">Cryptocurrency intelligence platform for fund monitoring and compliance</p>
                </div>
              </div>
              <div className="hidden md:block">
                <div className="flex gap-4">
                  <div className="p-3 rounded-lg bg-white/10 backdrop-blur-sm">
                    <Lock className="h-6 w-6 text-white" />
                  </div>
                  <div className="p-3 rounded-lg bg-white/10 backdrop-blur-sm">
                    <ShieldCheck className="h-6 w-6 text-white" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

          {/* Market Pulse - More valuable than generic stats */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Market Pulse</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {marketPulse.map((item, index) => (
                <div 
                  key={item.label} 
                  className="bg-white/70 backdrop-blur-sm rounded-xl border border-gray-100 p-4 shadow-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-3 rounded-lg ${
                      index === 0 ? 'bg-green-50' : 
                      index === 1 ? 'bg-blue-50' : 
                      index === 2 ? 'bg-purple-50' : 
                      'bg-red-50'
                    }`}>
                      <item.icon className={`h-5 w-5 ${
                        index === 0 ? 'text-green-500' : 
                        index === 1 ? 'text-blue-500' : 
                        index === 2 ? 'text-purple-500' : 
                        'text-red-500'
                      }`} />
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">{item.label}</div>
                      <div className="flex items-center gap-2">
                        <span className="text-xl font-semibold text-gray-800">
                          {item.value}
                        </span>
                        {item.change && (
                          <span className={`text-xs font-medium ${
                            item.trend === 'up' ? 'text-green-500' : 'text-red-500'
                          }`}>
                            {item.change}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Features Grid - Minimalist, Professional */}
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Platform Tools</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feature) => (
              <Link
                key={feature.label}
                href={feature.href}
                className="group"
              >
                <div className="bg-white/70 backdrop-blur-sm rounded-xl border border-gray-100 p-5 h-full hover:shadow-md transition-all flex flex-col">
                  <div className="flex items-start gap-4 mb-3">
                    <div className={`${feature.bgColor} p-3 rounded-xl`}>
                      <feature.icon className={`h-5 w-5 ${feature.color}`} />
                    </div>
                    <div>
                      <h3 className="text-lg font-medium text-gray-800 mb-1">{feature.label}</h3>
                      <p className="text-gray-500 text-sm">{feature.description}</p>
                    </div>
                  </div>
                  <div className="mt-auto pt-3 flex justify-end">
                    <div className="text-sm font-medium text-blue-600 group-hover:text-blue-700 flex items-center gap-1">
                      <span>Open</span>
                      <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
          
          {/* Centered Report Generator Feature */}
          <div className="mt-4 flex justify-center">
            <Link
              href={reportGeneratorFeature.href}
              className="group w-full sm:w-2/3 lg:w-1/3"
            >
              <div className="bg-white/70 backdrop-blur-sm rounded-xl border border-gray-100 p-5 h-full hover:shadow-md transition-all flex flex-col relative overflow-hidden">
                {/* Special background effect for the card */}
                <div className="absolute inset-0 bg-gradient-to-br from-amber-50 to-transparent opacity-40"></div>
                <div className="absolute top-0 right-0 h-20 w-20 bg-amber-100 rounded-bl-full opacity-60"></div>
                
                <div className="relative flex items-start gap-4 mb-3">
                  <div className={`${reportGeneratorFeature.bgColor} p-3 rounded-xl`}>
                    <reportGeneratorFeature.icon className={`h-5 w-5 ${reportGeneratorFeature.color}`} />
                  </div>
                  <div>
                    <div className="flex items-center">
                      <h3 className="text-lg font-medium text-gray-800 mb-1">{reportGeneratorFeature.label}</h3>
                      <div className="ml-2 px-2 py-0.5 bg-amber-100 text-amber-800 text-xs rounded-full flex items-center">
                        <Star className="h-3 w-3 mr-1" />
                        <span>Featured</span>
                      </div>
                    </div>
                    <p className="text-gray-500 text-sm">{reportGeneratorFeature.description}</p>
                  </div>
                </div>
                <div className="mt-auto pt-3 flex justify-end relative">
                  <div className="text-sm font-medium text-amber-700 group-hover:text-amber-800 flex items-center gap-1">
                    <span>Generate Reports</span>
                    <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
                  </div>
                </div>
              </div>
            </Link>
          </div>
          
         {/* Premium Banner */}
          <div className="mt-12 mb-6 relative">
            <div className="relative overflow-hidden rounded-2xl">
              <div className="absolute inset-0 bg-gradient-to-r from-[#1a1a3a] via-blue-600 to-[#4c6bff] opacity-90"></div>
              
              {/* Snake-like moving elements */}
              <div className="absolute top-0 left-0 w-full h-full overflow-hidden">
                {/* Snake body segments */}
                <div className="snake-segment absolute h-10 w-10 rounded-full bg-white/20 left-0 top-1/4"></div>
                <div className="snake-segment absolute h-9 w-9 rounded-full bg-white/20 left-0 top-1/4" style={{animationDelay: "0.1s"}}></div>
                <div className="snake-segment absolute h-8 w-8 rounded-full bg-white/20 left-0 top-1/4" style={{animationDelay: "0.2s"}}></div>
                <div className="snake-segment absolute h-7 w-7 rounded-full bg-white/20 left-0 top-1/4" style={{animationDelay: "0.3s"}}></div>
                <div className="snake-segment absolute h-6 w-6 rounded-full bg-white/20 left-0 top-1/4" style={{animationDelay: "0.4s"}}></div>
                <div className="snake-segment absolute h-5 w-5 rounded-full bg-white/20 left-0 top-1/4" style={{animationDelay: "0.5s"}}></div>
                <div className="snake-segment absolute h-4 w-4 rounded-full bg-white/20 left-0 top-1/4" style={{animationDelay: "0.6s"}}></div>
                
                {/* Second row of snake segments */}
                <div className="snake-segment-reverse absolute h-10 w-10 rounded-full bg-white/20 right-0 top-2/3"></div>
                <div className="snake-segment-reverse absolute h-9 w-9 rounded-full bg-white/20 right-0 top-2/3" style={{animationDelay: "0.1s"}}></div>
                <div className="snake-segment-reverse absolute h-8 w-8 rounded-full bg-white/20 right-0 top-2/3" style={{animationDelay: "0.2s"}}></div>
                <div className="snake-segment-reverse absolute h-7 w-7 rounded-full bg-white/20 right-0 top-2/3" style={{animationDelay: "0.3s"}}></div>
                <div className="snake-segment-reverse absolute h-6 w-6 rounded-full bg-white/20 right-0 top-2/3" style={{animationDelay: "0.4s"}}></div>
                <div className="snake-segment-reverse absolute h-5 w-5 rounded-full bg-white/20 right-0 top-2/3" style={{animationDelay: "0.5s"}}></div>
                <div className="snake-segment-reverse absolute h-4 w-4 rounded-full bg-white/20 right-0 top-2/3" style={{animationDelay: "0.6s"}}></div>
              </div>
              
              <div className="relative px-8 py-10 md:py-12 flex flex-col md:flex-row items-center justify-between">
                <div className="mb-6 md:mb-0 md:w-3/4">
                  <div className="flex items-center">
                    <Crown className="h-6 w-6 text-blue-200 mr-3" />
                    <h2 className="text-2xl md:text-3xl font-bold text-white">Unlock Premium Features</h2>
                  </div>
                  <p className="text-white/90 mt-3 max-w-2xl text-base md:text-lg">
                    Take your cryptocurrency compliance to the next level with our premium tools. 
                    Automate reports, get real-time alerts, and ensure regulatory compliance.
                  </p>
                  
                  <div className="mt-4 flex flex-wrap gap-4">
                    <div className="inline-flex items-center bg-[#1a1a3a]/50 backdrop-blur-md rounded-full px-4 py-1.5">
                      <Sparkles className="h-4 w-4 text-blue-200 mr-2" />
                      <span className="text-white text-sm font-medium">Advanced Report Templates</span>
                    </div>
                    <div className="inline-flex items-center bg-[#1a1a3a]/50 backdrop-blur-md rounded-full px-4 py-1.5">
                      <Sparkles className="h-4 w-4 text-blue-200 mr-2" />
                      <span className="text-white text-sm font-medium">Unlimited Document Analysis</span>
                    </div>
                    <div className="inline-flex items-center bg-[#1a1a3a]/50 backdrop-blur-md rounded-full px-4 py-1.5">
                      <Sparkles className="h-4 w-4 text-blue-200 mr-2" />
                      <span className="text-white text-sm font-medium">Priority Compliance Support</span>
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-col items-center">
                  <button className="bg-white text-[#1a1a3a] hover:bg-white/90 transition-colors py-3 px-8 rounded-xl font-semibold shadow-lg flex items-center group">
                    Upgrade Now 
                    <ChevronRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
                  </button>
                  <span className="text-white/80 text-xs mt-2">Try free for 14 days</span>
                </div>
              </div>
            </div>
            
            {/* Animated glow effect */}
            <div className="absolute -inset-1 bg-gradient-to-r from-[#1a1a3a] via-blue-600 to-[#4c6bff] rounded-2xl blur-xl opacity-30 animate-pulse-slow -z-10"></div>
          </div>
        </div>
      </div>
      
      {/* CSS for custom animations */}
      <style jsx global>{`
        @keyframes float-slow {
          0%, 100% { transform: translateY(0) translateX(0); }
          50% { transform: translateY(-20px) translateX(10px); }
        }
        @keyframes float-medium {
          0%, 100% { transform: translateY(0) translateX(0); }
          50% { transform: translateY(-15px) translateX(-15px); }
        }
        @keyframes float-fast {
          0%, 100% { transform: translateY(0) translateX(0); }
          50% { transform: translateY(10px) translateX(-10px); }
        }
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 0.5; }
        }
        @keyframes snake-move {
          0% { transform: translateX(-20px); }
          100% { transform: translateX(calc(100vw + 20px)); }
        }
        @keyframes snake-move-reverse {
          0% { transform: translateX(calc(100vw + 20px)); }
          100% { transform: translateX(-20px); }
        }
        .animate-float-slow {
          animation: float-slow 8s ease-in-out infinite;
        }
        .animate-float-medium {
          animation: float-medium 6s ease-in-out infinite;
        }
        .animate-float-fast {
          animation: float-fast 4s ease-in-out infinite;
        }
        .animate-pulse-slow {
          animation: pulse-slow 4s ease-in-out infinite;
        }
        .snake-segment {
          animation: snake-move 15s linear infinite;
        }
        .snake-segment-reverse {
          animation: snake-move-reverse 15s linear infinite;
        }
      `}</style>
    </div>
  );
}