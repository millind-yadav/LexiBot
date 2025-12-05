import React from 'react';
import { Upload } from 'lucide-react';

interface DocumentUploadProps {
  onUpload: (file: File) => void;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({ onUpload }) => {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onUpload(file);
    }
  };

  return (
    <div className="w-full max-w-md">
      <label 
        htmlFor="file-upload"
        className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors"
      >
        <div className="flex flex-col items-center justify-center pt-5 pb-6">
          <Upload className="w-8 h-8 mb-2 text-gray-500" />
          <p className="text-sm text-gray-500 font-medium">Click to upload contract</p>
          <p className="text-xs text-gray-400">PDF, DOCX (MAX. 10MB)</p>
        </div>
        <input 
          id="file-upload" 
          type="file" 
          className="hidden" 
          onChange={handleFileChange}
          accept=".pdf,.docx,.txt"
        />
      </label>
    </div>
  );
};
