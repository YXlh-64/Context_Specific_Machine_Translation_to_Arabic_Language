import { Card, CardContent } from '@/components/ui/card';

interface SkeletonCardProps {
  rank: number;
}

export const SkeletonCard = ({ rank }: SkeletonCardProps) => {
  return (
    <Card className="border border-border bg-card overflow-hidden">
      <CardContent className="p-0">
        {/* Card Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-border/50">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded skeleton-shimmer" />
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Variation {rank}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-7 h-7 rounded skeleton-shimmer" />
            <div className="w-7 h-7 rounded skeleton-shimmer" />
          </div>
        </div>

        {/* Card Body - Skeleton lines */}
        <div className="p-4 space-y-3" dir="rtl">
          <div className="h-5 w-full rounded skeleton-shimmer" />
          <div className="h-5 w-4/5 rounded skeleton-shimmer ml-auto" />
          <div className="h-5 w-3/5 rounded skeleton-shimmer ml-auto" />
        </div>
      </CardContent>
    </Card>
  );
};
