"use client";
import { useEffect, useState, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";

export default function ImageSearchPage() {
  const [description, setDescription] = useState("");
  const [images, setImages] = useState([]);
  const [visibleCount, setVisibleCount] = useState(25);
  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);

  const fetchImages = async () => {
    const res = await fetch(`http://localhost:8000/search?description=${description}`);
    const data = await res.json();
    setImages(data.images);
    setVisibleCount(25);
  };

  useEffect(() => {
    fetchImages();
  }, []);

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    await fetch("http://localhost:8000/upload", { method: "POST", body: formData });
    fetchImages();
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) setFile(selectedFile);
  };

  const resetFile = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = null;
    }
  };

  return (
    <div className="min-h-screen bg-black">
      <motion.div
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="bg-black h-[30vh] p-6 flex flex-col justify-center items-center gap-4 shadow-md rounded-b-2xl"
      >
        <h1 className="text-white text-4xl font-bold mb-2 tracking-wide">Gallery Worded</h1>
        <div className="flex flex-wrap items-center gap-4 justify-center">
          <Input
            placeholder="Search images..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="bg-white text-black rounded-xl px-4 py-2 w-64 shadow-sm"
          />
          <Button
            onClick={fetchImages}
            className="bg-white text-black font-semibold hover:bg-gray-200"
          >
            Search
          </Button>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="p-6 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6 bg-black"
      >
        {images.slice(0, visibleCount).map((img, idx) => (
          <motion.img
            key={idx}
            src={`http://localhost:8000/images/${img}`}
            alt="result"
            whileHover={{ scale: 1.05 }}
            className="w-full rounded-2xl shadow-lg object-cover aspect-square"
          />
        ))}
      </motion.div>

      {visibleCount < images.length && (
        <div className="flex justify-center mt-6">
          <Button
            onClick={() => setVisibleCount((prev) => prev + 25)}
            className="bg-white text-black font-medium px-6 py-2 rounded-xl hover:bg-gray-200"
          >
            Load More
          </Button>
        </div>
      )}
    </div>
  );
}
