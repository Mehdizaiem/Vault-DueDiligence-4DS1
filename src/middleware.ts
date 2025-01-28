import { authMiddleware } from "@clerk/nextjs";
 
export default authMiddleware({
  // Routes that can be accessed while signed out
  publicRoutes: ["/sign-in", "/sign-up"],
  // Routes that can always be accessed, and have no authentication information
  ignoredRoutes: ["/api/webhook"],
  afterAuth(auth, req ) {
    // Handle users who aren't authenticated
    if (!auth.userId && !auth.isPublicRoute) {
      return Response.redirect(new URL('/sign-in', req.url));
    }
    // If the user is logged in and trying to access a public route,
    // redirect them to the dashboard
    if (auth.userId && ["/", "/sign-in", "/sign-up"].includes(req.nextUrl.pathname)) {
      return Response.redirect(new URL('/dashboard', req.url));
    }
  }
});

export const config = {
  matcher: ["/((?!.+\\.[\\w]+$|_next).*)", "/", "/(api|trpc)(.*)"],
};